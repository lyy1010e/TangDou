import os
import json
import threading
import time
import shutil
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

from .config import load_config
from .cover import download_cover, resolve_cover_url
from .downloader import DownloadOutcome, download_file_with_retry
from .enhancer import enhance_video_to_1080p, trim_video
from .source_selector import SourceChoice, collect_source_candidates, select_best_source, source_height
from .url_signer import build_page_url
from .utils import clean_filename, thread_safe_print


HEADERS = {
    'Referer': 'http://www.tangdou.com/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
}
DOWNLOAD_WORKERS = 3
FFMPEG_ENHANCE_WORKERS = 2
REALESRGAN_ENHANCE_WORKERS = 1
PREFER_HIGHEST_SOURCE = True
TANGDOU_HEAD_EXTRA_SECONDS = 2.0


@dataclass(frozen=True)
class VideoDownloadResult:
    success: bool
    fail_info: Optional[str]
    fail_reason: Optional[str]
    video_info: str
    data: dict
    play_data: Optional[dict] = None
    source_choice: Optional[SourceChoice] = None
    outcome: Optional[DownloadOutcome] = None


def load_urls_from_log():
    """从 urls.txt 加载 URL 列表。"""
    urls = []
    script_dir = Path(__file__).resolve().parent.parent
    log_path = script_dir / 'urls.txt'
    if not log_path.exists() or not log_path.is_file():
        print('[提示] 未找到 urls.txt')
        print(f'[提示] 脚本目录: {script_dir}')
        return urls

    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    for encoding in encodings:
        try:
            line_count = 0
            urls.clear()
            with open(log_path, 'r', encoding=encoding) as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line and not line.startswith('http'):
                        _, url = line.split('=', 1)
                        line = url.strip()
                    if line.startswith('http'):
                        urls.append(line)

            if urls:
                print(f'[提示] 成功从文件读取URL: {log_path.name} (编码: {encoding}, 总行数: {line_count}, URL数: {len(urls)})')
                break
            if line_count > 0:
                print(f'[警告] 文件 {log_path.name} 有 {line_count} 行，但未找到有效URL')
        except Exception as e:
            print(f'[错误] 读取文件 {log_path} 失败 (编码: {encoding}): {e}')

    return urls


def _response_error_text(response_data):
    code = response_data.get('code')
    msg = response_data.get('msg') or '未知错误'
    return f'code={code}, msg={msg}'


def collect_all_videos(download_urls, debug_api_response=False):
    """收集所有待下载的视频信息。"""
    all_videos = []

    print(f'\n{"=" * 60}')
    print('[阶段1] 收集所有待下载视频信息')
    print(f'{"=" * 60}')

    global_num = 1
    for template_idx, template_url in enumerate(download_urls, 1):
        page_idx = 1
        while True:
            collect_url = build_page_url(template_url, page_idx)
            try:
                print(f'\n[收集] 模板{template_idx} 正在处理第{page_idx}页...')
                response = requests.get(url=collect_url, headers=HEADERS, timeout=(30, 30))
                response.raise_for_status()
                response_data = response.json()

                response_code = response_data.get('code')
                has_error_code = response_code is not None and str(response_code) != '0'
                if has_error_code:
                    print(f'[警告] 模板{template_idx} 第{page_idx}页接口返回异常: {_response_error_text(response_data)}')
                    break

                datas = response_data.get('datas', [])
                if not isinstance(datas, list):
                    datas = []

                page_size = response_data.get('pagesize', len(datas))
                if not datas:
                    print(f'[收集] 模板{template_idx} 第{page_idx}页无数据，自动翻页结束')
                    break

                print(f'[收集] 模板{template_idx} 第{page_idx}页: 找到 {len(datas)} 个视频 (pagesize: {page_size})')

                for index, data in enumerate(datas):
                    if debug_api_response:
                        print(f'[调试] 模板{template_idx} 第{page_idx}页-第{index + 1}个 列表接口视频条目:')
                        print(json.dumps(data, ensure_ascii=False, indent=2))
                    all_videos.append({
                        'data': data,
                        'num': global_num,
                        'page_info': f'模板{template_idx}-第{page_idx}页-第{index + 1}个',
                    })
                    global_num += 1

                page_idx += 1
                time.sleep(0.3)
            except Exception as e:
                print(f'[错误] 处理模板{template_idx} 第{page_idx}页失败: {e}')
                break

    print(f'\n[收集完成] 共收集到 {len(all_videos)} 个待下载视频')
    print(f'{"=" * 60}\n')
    return all_videos


def _update_counter(counter, lock, key):
    with lock:
        counter[key] += 1


def _add_counter(counter, lock, key, value):
    with lock:
        counter[key] += value


def _to_float(value, default=0.0):
    try:
        if value in (None, ''):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _video_upscale_config(base_config, data):
    head_t = _to_float(data.get('head_t'), 0.0)
    end_t = _to_float(data.get('end_t'), 0.0)
    duration = _to_float(data.get('duration'), 0.0)
    trim_head = head_t + TANGDOU_HEAD_EXTRA_SECONDS if head_t > 0 else 0.0
    trim_duration = max(duration - trim_head - end_t, 0.0) if duration > 0 else 0.0
    return replace(
        base_config,
        trim_head_seconds=trim_head,
        trim_duration_seconds=trim_duration,
    )


def _enhance_worker_count(config):
    if config.upscale.enhance_engine == 'realesrgan':
        return REALESRGAN_ENHANCE_WORKERS
    return FFMPEG_ENHANCE_WORKERS


def _has_trim_markers(data):
    return _to_float(data.get('head_t'), 0.0) > 0 or _to_float(data.get('end_t'), 0.0) > 0


def _source_height_text(source_size):
    return f'{source_size}p' if source_size else '未知'


def _postprocess_mode(download_result, upscale_config):
    is_new_download = download_result.success and download_result.outcome and download_result.outcome.downloaded
    if not is_new_download or not download_result.source_choice:
        return None

    enhance_enabled = upscale_config.enabled
    trim_enabled = upscale_config.trim_enabled
    if not enhance_enabled and not trim_enabled:
        return None

    source_size = source_height(download_result.source_choice)
    is_low_source = source_size == 0 or source_size < upscale_config.target_height
    has_trim = _has_trim_markers(download_result.data)

    # 低分辨率源且开启了增强：走增强流程（增强时也会一并裁掉片头片尾）。
    if is_low_source and enhance_enabled:
        return 'enhance'
    # 有片头片尾标记且开启了裁剪：只裁剪，保留原分辨率。
    # 包括高分辨率源，或低分辨率源但未开启增强的情况。
    if has_trim and trim_enabled:
        return 'trim'
    # 开启了去片头片尾但没有标记：直接复制到 enhanced 目录，保证两边数量一致。
    if trim_enabled:
        return 'copy'
    return None


def _handle_postprocess(download_result, mode, config, counter, lock):
    if not mode or not download_result.outcome or not download_result.source_choice:
        return

    source_size = source_height(download_result.source_choice)
    if mode == 'copy':
        action_text = '复制'
        counter_prefix = 'copy'
        target_text = '原分辨率'
        try:
            input_path = download_result.outcome.filepath
            output_height = source_size or config.upscale.target_height
            output_path = Path(config.upscale.output_dir) / f'{input_path.stem}_{output_height}p.mp4'
            cover_url = resolve_cover_url(download_result.data, download_result.play_data)
            cover_path = download_cover(
                cover_url,
                Path(config.upscale.output_dir) / f'{download_result.outcome.filepath.stem}_cover.jpg',
                HEADERS,
            )
            thread_safe_print(
                f'[{action_text}] {download_result.video_info} 源:{_source_height_text(source_size)} -> {target_text} | '
                f'无片头片尾标记，直接复制'
            )
            start = time.time()
            shutil.copy2(input_path, output_path)
            elapsed = time.time() - start
            thread_safe_print(f'[{action_text}] {download_result.video_info} 复制完成 -> {output_path.name} | 耗时:{elapsed:.1f}秒')
            _update_counter(counter, lock, f'{counter_prefix}_success')
            _add_counter(counter, lock, f'{counter_prefix}_seconds', elapsed)
        except Exception as e:
            _update_counter(counter, lock, f'{counter_prefix}_fail')
            thread_safe_print(f'[{action_text}失败] {download_result.video_info}: {e}')
        return

    action_text = '增强' if mode == 'enhance' else '裁剪'
    counter_prefix = 'upscale' if mode == 'enhance' else 'trim'
    target_text = f'{config.upscale.target_height}p' if mode == 'enhance' else '原分辨率'
    try:
        upscale_config = _video_upscale_config(config.upscale, download_result.data)
        cover_url = resolve_cover_url(download_result.data, download_result.play_data)
        cover_path = download_cover(
            cover_url,
            Path(config.upscale.output_dir) / f'{download_result.outcome.filepath.stem}_cover.jpg',
            HEADERS,
        )
        thread_safe_print(
            f'[{action_text}] {download_result.video_info} 源:{_source_height_text(source_size)} -> {target_text} | '
            f'引擎:{config.upscale.enhance_engine if mode == "enhance" else "ffmpeg"} | '
            f'片头:{upscale_config.trim_head_seconds:g}秒'
        )
        if mode == 'enhance':
            upscale_result = enhance_video_to_1080p(download_result.outcome.filepath, upscale_config, config.tools, cover_path=cover_path)
        else:
            upscale_result = trim_video(
                download_result.outcome.filepath,
                upscale_config,
                config.tools,
                cover_path=cover_path,
                source_height=source_size,
            )
        thread_safe_print(f'[{action_text}] {download_result.video_info} {upscale_result.message} | 耗时:{upscale_result.elapsed_seconds:.1f}秒')
        counter_key = f'{counter_prefix}_skip' if upscale_result.skipped else f'{counter_prefix}_success'
        _update_counter(counter, lock, counter_key)
        _add_counter(counter, lock, f'{counter_prefix}_seconds', upscale_result.elapsed_seconds)
    except Exception as e:
        _update_counter(counter, lock, f'{counter_prefix}_fail')
        thread_safe_print(f'[{action_text}失败] {download_result.video_info}: {e}')


def _print_play_response_debug(video_info, video_data):
    print(f'[调试] {video_info} 播放接口原始返回:')
    print(json.dumps(video_data, ensure_ascii=False, indent=2))
    print(f'[调试] {video_info} 识别到的视频源候选:')
    candidates = collect_source_candidates(video_data)
    if not candidates:
        print('  - 未识别到候选视频源')
        return
    for candidate in candidates:
        url_text = candidate.url
        if len(url_text) > 160:
            url_text = f'{url_text[:157]}...'
        hd_text = '高清候选' if candidate.is_hd else '普通候选'
        print(f'  - {hd_text} | height={_source_height_text(source_height(candidate))} | score={candidate.score} | {candidate.label} | {url_text}')


def _play_error_reason(video_data, fallback):
    if not isinstance(video_data, dict):
        return fallback
    code = video_data.get('code')
    msg = video_data.get('msg')
    if msg:
        return f'播放接口返回: {msg}'
    if code not in (None, 0, '0'):
        return f'播放接口返回异常 code={code}'
    return fallback


def download_single_video(data, num, page_size, download_dir, config, lock, retry_info_list, retry_lock):
    """下载单个视频，返回后续增强所需的信息。"""
    title = data.get('title', '未知')
    video_info = f'{num} - {title}'
    try:
        vid = data['vid']
        play_info = f'https://api-h5.tangdou.com/mtangdou/video/play?vid={vid}&page=1&uuid='
        video_data = requests.get(url=play_info, headers=HEADERS, timeout=(30, 30)).json()
        thread_safe_print(f'\n[{num}/{page_size}] 开始下载: {title}')
        if config.download.debug_api_response:
            with lock:
                _print_play_response_debug(video_info, video_data)

        source_choice = select_best_source(video_data, PREFER_HIGHEST_SOURCE)
        if not source_choice.url:
            raise ValueError(_play_error_reason(video_data, source_choice.label))

        source_text = '高清源' if source_choice.is_hd else '普通源'
        source_size = source_height(source_choice)
        thread_safe_print(f'[源选择] {video_info} 使用{source_text}: {source_choice.label} | 源:{_source_height_text(source_size)}')

        safe_title = clean_filename(title)
        filepath = download_dir / f'{num}_{safe_title}.mp4'
        outcome = download_file_with_retry(
            source_choice.url,
            filepath,
            HEADERS,
            video_info=video_info,
            retry_info_list=retry_info_list,
            retry_lock=retry_lock,
        )

        if outcome.success:
            thread_safe_print(f'✓ 下载成功: {video_info}')
            return VideoDownloadResult(True, None, None, video_info, data, video_data, source_choice, outcome)

        fail_reason = '下载文件失败，可能是源地址失效或网络分段无进展'
        thread_safe_print(f'✗ 下载失败: {video_info} | {fail_reason}')
        return VideoDownloadResult(False, video_info, fail_reason, video_info, data, video_data, source_choice, outcome)
    except Exception as e:
        fail_reason = str(e) or type(e).__name__
        thread_safe_print(f'===》下载出错 [{num}]: {fail_reason}')
        return VideoDownloadResult(False, video_info, fail_reason, video_info, data)


def process_single_video(download_result, mode, config, counter, lock):
    if not download_result.success:
        return
    _handle_postprocess(download_result, mode, config, counter, lock)


def _print_retry_info(retry_info_list):
    if not retry_info_list:
        return

    print(f'\n[发生过分段续传的视频] 共 {len(retry_info_list)} 个')
    for retry_info in sorted(retry_info_list, key=lambda x: x.get('retry_count', 0), reverse=True):
        print(
            f"  - {retry_info.get('video_info', '未知')} | "
            f"最终{retry_info.get('final_status', '已继续下载')} | "
            f"{retry_info.get('error_type', '未知')} | "
            f"重试{retry_info.get('retry_count', 0)}次 | "
            f"断点位置{retry_info.get('downloaded_mb', 0):.1f}MB"
        )


def _clear_download_dir(download_dir):
    """启动前清空 download 目录内容，避免旧文件影响本次下载和增强判断。"""
    if not download_dir.exists():
        return
    if download_dir.resolve().name.lower() != 'download':
        print(f'[警告] 输出目录不是 download，已跳过自动清理: {download_dir}')
        return

    removed_count = 0
    for item in download_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
        removed_count += 1

    if removed_count:
        print(f'[清理] download 目录已有内容，已删除 {removed_count} 项')


def download_video():
    config = load_config()
    download_urls = load_urls_from_log()
    if not download_urls:
        print('[提示] 未找到 URL，请检查 urls.txt 文件')
        return

    print(f'[提示] 从 urls.txt 加载了 {len(download_urls)} 个URL')
    all_videos = collect_all_videos(download_urls, config.download.debug_api_response)
    if not all_videos:
        print('[错误] 未找到任何待下载的视频')
        return

    download_dir = Path(config.download.output_dir)
    _clear_download_dir(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    total_count = len(all_videos)
    start_time = time.time()
    counter = {
        'upscale_success': 0,
        'upscale_skip': 0,
        'upscale_fail': 0,
        'upscale_seconds': 0.0,
        'trim_success': 0,
        'trim_skip': 0,
        'trim_fail': 0,
        'trim_seconds': 0.0,
        'copy_success': 0,
        'copy_fail': 0,
        'copy_seconds': 0.0,
    }
    failed_files = []
    download_results = []
    retry_info_list = []
    lock = threading.Lock()

    print(f'\n{"=" * 60}')
    print('[阶段2] 开始下载所有视频')
    print(f'[统计] 总视频数: {total_count}')
    print(f'{"=" * 60}\n')
    print(f'[配置] 下载并发: {DOWNLOAD_WORKERS} (分段续传策略：兼顾速度和稳定性)')

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        future_info = {}
        for video_info in all_videos:
            data = video_info['data']
            num = video_info['num']
            page_info = video_info['page_info']
            future = executor.submit(
                download_single_video,
                data,
                num,
                total_count,
                download_dir,
                config,
                lock,
                retry_info_list,
                lock,
            )
            future_info[future] = (num, data.get('title', '未知'), page_info)

        completed = 0
        completed_success = 0
        completed_fail = 0
        for future in as_completed(future_info):
            num, title, page_info = future_info[future]
            try:
                result = future.result()
                download_results.append(result)
                completed += 1
                completed_success += 1 if result.success else 0
                completed_fail += 0 if result.success else 1
                if not result.success and result.fail_info:
                    reason_text = f' | 原因: {result.fail_reason}' if result.fail_reason else ''
                    failed_files.append(f'{result.fail_info} ({page_info}){reason_text}')
                if completed % 5 == 0 or completed == total_count:
                    print(f'[进度] 已完成: {completed}/{total_count} | 成功: {completed_success} | 失败: {completed_fail}')
            except Exception as e:
                completed += 1
                completed_fail += 1
                failed_files.append(f'{num} - {title} ({page_info})')
                print(f'[错误] 任务执行异常 [{num}] ({page_info}): {e}')

    postprocess_plans = [
        (result, mode)
        for result in download_results
        for mode in [_postprocess_mode(result, config.upscale)]
        if mode
    ]
    any_postprocess_enabled = config.upscale.enabled or config.upscale.trim_enabled
    if postprocess_plans:
        enhance_workers = _enhance_worker_count(config)
        enhance_count = sum(1 for _, mode in postprocess_plans if mode == 'enhance')
        trim_count = sum(1 for _, mode in postprocess_plans if mode == 'trim')
        copy_count = sum(1 for _, mode in postprocess_plans if mode == 'copy')
        print(f'\n{"=" * 60}')
        print('[阶段3] 开始处理新下载视频')
        print(f'[配置] 增强:{("开" if config.upscale.enabled else "关")} | 去片头片尾:{("开" if config.upscale.trim_enabled else "关")} | 目标: {config.upscale.target_height}p | 并发: {enhance_workers}')
        print(f'[统计] 增强: {enhance_count} 个 | 只裁剪: {trim_count} 个 | 直接复制: {copy_count} 个')
        print(f'{"=" * 60}\n')

        with ThreadPoolExecutor(max_workers=enhance_workers) as executor:
            future_info = {
                executor.submit(process_single_video, result, mode, config, counter, lock): (result, mode)
                for result, mode in postprocess_plans
            }
            process_completed = 0
            for future in as_completed(future_info):
                result, mode = future_info[future]
                if mode == 'enhance':
                    counter_prefix = 'upscale'
                    action_text = '增强'
                elif mode == 'trim':
                    counter_prefix = 'trim'
                    action_text = '裁剪'
                else:
                    counter_prefix = 'copy'
                    action_text = '复制'
                try:
                    future.result()
                except Exception as e:
                    _update_counter(counter, lock, f'{counter_prefix}_fail')
                    print(f'[{action_text}失败] {result.video_info}: {e}')
                process_completed += 1
                if process_completed % 5 == 0 or process_completed == len(postprocess_plans):
                    print(f'[处理进度] 已处理: {process_completed}/{len(postprocess_plans)}')
    elif not any_postprocess_enabled:
        print('\n[处理] 增强和去片头片尾均已关闭，跳过视频处理阶段')
    else:
        print('\n[处理] 没有需要增强、裁剪或复制的新下载视频')

    total_time = time.time() - start_time
    _print_retry_info(retry_info_list)

    if failed_files:
        print('\n[失败文件列表]')
        for fail_file in failed_files:
            print(f'  ✗ {fail_file}')

    print(f'\n{"=" * 60}')
    print('[统计] 任务完成')
    print(f'  总视频数: {total_count}')
    print(f'  成功: {completed_success}')
    print(f'  失败: {completed_fail}')
    print(f'  视频增强成功: {counter["upscale_success"]}')
    print(f'  视频增强跳过: {counter["upscale_skip"]}')
    print(f'  视频增强失败: {counter["upscale_fail"]}')
    print(f'  视频裁剪成功: {counter["trim_success"]}')
    print(f'  视频裁剪跳过: {counter["trim_skip"]}')
    print(f'  视频裁剪失败: {counter["trim_fail"]}')
    print(f'  视频直接复制: {counter["copy_success"]}')
    print(f'  视频复制失败: {counter["copy_fail"]}')
    print(f'  成功率: {(completed_success / total_count * 100):.1f}%' if total_count > 0 else '  成功率: 0%')
    print(f'  总耗时: {total_time:.1f}秒')
    if completed > 0:
        print(f'  平均耗时: {total_time / completed:.1f}秒/个')
    if counter['upscale_success'] > 0:
        print(f'  增强耗时: {counter["upscale_seconds"]:.1f}秒')
    if counter['trim_success'] > 0:
        print(f'  裁剪耗时: {counter["trim_seconds"]:.1f}秒')
    print(f'{"=" * 60}')
