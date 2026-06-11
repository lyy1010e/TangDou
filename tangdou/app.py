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
from .downloader import download_file_with_retry
from .enhancer import enhance_video_to_1080p
from .source_selector import collect_source_candidates, select_best_source, source_height
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
    video_info: str
    data: dict
    play_data: Optional[dict] = None
    source_choice: object = None
    outcome: object = None


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


def collect_all_videos(download_urls, debug_api_response=False):
    """收集所有待下载的视频信息。"""
    all_videos = []

    print(f'\n{"=" * 60}')
    print('[阶段1] 收集所有待下载视频信息')
    print(f'{"=" * 60}')

    global_num = 1
    for page_idx, collect_url in enumerate(download_urls, 1):
        try:
            print(f'\n[收集] 正在处理第{page_idx}页URL...')
            response = requests.get(url=collect_url, headers=HEADERS, timeout=(30, 30))
            response.raise_for_status()
            response_data = response.json()

            datas = response_data.get('datas', [])
            page_size = response_data.get('pagesize', len(datas))
            print(f'[收集] 第{page_idx}页: 找到 {len(datas)} 个视频 (pagesize: {page_size})')

            for index, data in enumerate(datas):
                if debug_api_response:
                    print(f'[调试] 第{page_idx}页-第{index + 1}个 列表接口视频条目:')
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                all_videos.append({
                    'data': data,
                    'num': global_num,
                    'page_info': f'第{page_idx}页-第{index + 1}个',
                })
                global_num += 1

            time.sleep(0.3)
        except Exception as e:
            print(f'[错误] 处理第{page_idx}页URL失败: {e}')

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


def _should_enhance_download_result(download_result, upscale_config):
    is_new_download = download_result.success and download_result.outcome and download_result.outcome.downloaded
    if not is_new_download or not download_result.source_choice:
        return False

    source_size = source_height(download_result.source_choice)
    is_low_source = source_size == 0 or source_size < upscale_config.target_height
    return not download_result.source_choice.is_hd and is_low_source


def _handle_upscale(outcome, source_choice, config, counter, lock, video_info, data, play_data):
    source_size = source_height(source_choice)
    is_low_source = source_size == 0 or source_size < config.upscale.target_height
    should_upscale = outcome.success and outcome.downloaded and config.upscale.enabled and not source_choice.is_hd and is_low_source
    if not should_upscale:
        return

    try:
        upscale_config = _video_upscale_config(config.upscale, data)
        cover_url = resolve_cover_url(data, play_data)
        cover_path = download_cover(
            cover_url,
            Path(config.upscale.output_dir) / f'{outcome.filepath.stem}_cover.jpg',
            HEADERS,
        )
        thread_safe_print(
            f'[增强] {video_info} 普通源下载完成，开始使用 {config.upscale.enhance_engine} '
            f'生成 {config.upscale.target_height}p 增强版，裁剪片头 {upscale_config.trim_head_seconds:g} 秒'
        )
        upscale_result = enhance_video_to_1080p(outcome.filepath, upscale_config, config.tools, cover_path=cover_path)
        thread_safe_print(f'[增强] {video_info} {upscale_result.message}')
        counter_key = 'upscale_skip' if upscale_result.skipped else 'upscale_success'
        _update_counter(counter, lock, counter_key)
        _add_counter(counter, lock, 'upscale_seconds', upscale_result.elapsed_seconds)
    except Exception as e:
        _update_counter(counter, lock, 'upscale_fail')
        thread_safe_print(f'[增强失败] {video_info}: {e}')


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
        print(f'  - {hd_text} | score={candidate.score} | {candidate.label} | {url_text}')


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
            raise ValueError(source_choice.label)

        source_text = '高清源' if source_choice.is_hd else '普通源'
        thread_safe_print(f'[源选择] {video_info} 使用{source_text}: {source_choice.label}')

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
            return VideoDownloadResult(True, None, video_info, data, video_data, source_choice, outcome)

        thread_safe_print(f'✗ 下载失败: {video_info}')
        return VideoDownloadResult(False, video_info, video_info, data, video_data, source_choice, outcome)
    except Exception as e:
        thread_safe_print(f'===》下载出错 [{num}]: {e}')
        return VideoDownloadResult(False, video_info, video_info, data)


def enhance_single_video(download_result, config, counter, lock):
    if not download_result.success:
        return
    _handle_upscale(
        download_result.outcome,
        download_result.source_choice,
        config,
        counter,
        lock,
        download_result.video_info,
        download_result.data,
        download_result.play_data,
    )


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
                    failed_files.append(f'{result.fail_info} ({page_info})')
                if completed % 5 == 0 or completed == total_count:
                    print(f'[进度] 已完成: {completed}/{total_count} | 成功: {completed_success} | 失败: {completed_fail}')
            except Exception as e:
                completed += 1
                completed_fail += 1
                failed_files.append(f'{num} - {title} ({page_info})')
                print(f'[错误] 任务执行异常 [{num}] ({page_info}): {e}')

    enhance_candidates = [
        result
        for result in download_results
        if _should_enhance_download_result(result, config.upscale)
    ]
    if config.upscale.enabled and enhance_candidates:
        enhance_workers = _enhance_worker_count(config)
        print(f'\n{"=" * 60}')
        print('[阶段3] 开始增强新下载的普通源视频')
        print(f'[配置] 增强方式: {config.upscale.enhance_engine} | 目标: {config.upscale.target_height}p | 增强并发: {enhance_workers}')
        print(f'{"=" * 60}\n')

        with ThreadPoolExecutor(max_workers=enhance_workers) as executor:
            future_info = {
                executor.submit(enhance_single_video, result, config, counter, lock): result
                for result in enhance_candidates
            }
            enhance_completed = 0
            for future in as_completed(future_info):
                result = future_info[future]
                try:
                    future.result()
                except Exception as e:
                    _update_counter(counter, lock, 'upscale_fail')
                    print(f'[增强失败] {result.video_info}: {e}')
                enhance_completed += 1
                if enhance_completed % 5 == 0 or enhance_completed == len(enhance_candidates):
                    print(f'[增强进度] 已处理: {enhance_completed}/{len(enhance_candidates)}')
    elif not config.upscale.enabled:
        print('\n[增强] 配置已关闭，跳过视频增强阶段')
    else:
        print('\n[增强] 没有需要增强的新下载普通源视频')

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
    print(f'  成功率: {(completed_success / total_count * 100):.1f}%' if total_count > 0 else '  成功率: 0%')
    print(f'  总耗时: {total_time:.1f}秒')
    if completed > 0:
        print(f'  平均耗时: {total_time / completed:.1f}秒/个')
    if counter['upscale_success'] > 0:
        print(f'  增强耗时: {counter["upscale_seconds"]:.1f}秒')
    print(f'{"=" * 60}')
