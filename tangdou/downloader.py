import time
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.exceptions import ChunkedEncodingError, RequestException

from .utils import verify_mp4_file


MIN_SEGMENT_SIZE = 1024 * 1024
DEFAULT_SEGMENT_SIZE = 2 * 1024 * 1024


@dataclass(frozen=True)
class DownloadOutcome:
    success: bool
    downloaded: bool
    filepath: Path


def _merge_retry_info(retry_info_list, retry_lock, info):
    """新增或更新某个视频的重试统计。"""
    if retry_info_list is None:
        return

    def update():
        increment_retry = info.pop('_increment_retry', False)
        existing = next((r for r in retry_info_list if r.get('video_info') == info['video_info']), None)
        if existing:
            if increment_retry:
                info['retry_count'] = existing.get('retry_count', 0) + info.get('retry_count', 1)
            existing.update(info)
        else:
            retry_info_list.append(info)

    if retry_lock:
        with retry_lock:
            update()
    else:
        update()


def _mark_retry_status(retry_info_list, retry_lock, video_info, status):
    """只给已经发生过续传的视频补充最终状态。"""
    if retry_info_list is None:
        return

    def update():
        existing = next((r for r in retry_info_list if r.get('video_info') == video_info), None)
        if existing:
            existing['final_status'] = status

    if retry_lock:
        with retry_lock:
            update()
    else:
        update()


def _request_headers(headers, resume_pos=0, end_pos=None):
    result = headers.copy()
    result.update({
        'Accept': '*/*',
        'Accept-Encoding': 'identity',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'close',
    })
    if resume_pos or end_pos is not None:
        end_text = '' if end_pos is None else str(end_pos)
        result['Range'] = f'bytes={resume_pos}-{end_text}'
    return result


def _remote_size(url, headers):
    try:
        response = requests.head(url, headers=_request_headers(headers), timeout=(10, 30), allow_redirects=True)
        response.raise_for_status()
        size = int(response.headers.get('content-length') or 0)
        response.close()
        return size
    except Exception:
        return 0


def _chunk_size(total_size):
    if total_size > 50 * 1024 * 1024:
        return 128 * 1024
    if total_size > 10 * 1024 * 1024:
        return 64 * 1024
    return 32 * 1024


def _replace_with_retry(src, dst, video_prefix='', attempts=8):
    """Windows 偶发占用 .part 文件时，稍等后再完成改名。"""
    for attempt in range(1, attempts + 1):
        try:
            src.replace(dst)
            return True
        except OSError as e:
            if getattr(e, 'winerror', None) != 32 or attempt == attempts:
                print(f'\n[失败] {video_prefix}文件改名失败: {e}')
                return False
            wait_time = min(0.5 * attempt, 3)
            print(f'\n[等待] {video_prefix}.part 文件暂时被占用，{wait_time:.1f}秒后重试改名')
            time.sleep(wait_time)
    return False


def _failure(filepath):
    return DownloadOutcome(success=False, downloaded=False, filepath=Path(filepath))


def download_file_with_retry(
    url,
    filepath,
    headers,
    max_retries=None,
    chunk_size=256 * 1024,
    video_info=None,
    retry_info_list=None,
    retry_lock=None,
):
    """按固定 Range 小段下载：写入 .part，失败保留，下一次从断点继续。"""
    filepath = Path(filepath)
    part_path = filepath.with_suffix(filepath.suffix + '.part')
    total_size = _remote_size(url, headers)
    max_retries = max_retries or (15 if total_size > 100 * 1024 * 1024 else 12 if total_size > 50 * 1024 * 1024 else 10)
    segment_size = DEFAULT_SEGMENT_SIZE
    read_size = min(_chunk_size(total_size), 64 * 1024) if total_size else chunk_size
    video_prefix = f'[{video_info}] ' if video_info else ''

    if filepath.exists() and total_size and filepath.stat().st_size == total_size and verify_mp4_file(filepath):
        print(f'[跳过] {video_prefix}文件已完整存在')
        return DownloadOutcome(success=True, downloaded=False, filepath=filepath)

    if filepath.exists():
        filepath.unlink()
        print(f'[日志] {video_prefix}发现不完整成品文件，已删除并改用 .part 重新下载')

    stalled_retries = 0
    segment_count = 0
    last_log = time.time()

    while True:
        start_pos = part_path.stat().st_size if part_path.exists() else 0
        if total_size and start_pos >= total_size:
            break

        end_pos = min(start_pos + segment_size - 1, total_size - 1) if total_size else None
        segment_count += 1

        try:
            if start_pos == 0 and segment_count == 1:
                size_text = f'，大小 {total_size / 1024 / 1024:.1f}MB' if total_size else ''
                print(f'[下载] {video_prefix}开始分段下载{size_text}')

            with requests.get(
                url,
                headers=_request_headers(headers, start_pos, end_pos),
                stream=True,
                timeout=(30, 60),
                allow_redirects=True,
            ) as response:
                if total_size and response.status_code != 206:
                    raise ValueError(f'服务器未按 Range 返回分段，状态码: {response.status_code}')

                response.raise_for_status()
                content_range = response.headers.get('Content-Range')
                if content_range:
                    total_size = int(content_range.rsplit('/', 1)[-1])
                elif not total_size:
                    content_length = int(response.headers.get('content-length') or 0)
                    total_size = start_pos + content_length if content_length else 0

                mode = 'ab' if start_pos else 'wb'
                with open(part_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=read_size):
                        if chunk:
                            f.write(chunk)

            current_pos = part_path.stat().st_size if part_path.exists() else 0
            if current_pos <= start_pos:
                raise ChunkedEncodingError(f'分段无进展: {start_pos}')
            if total_size and current_pos > total_size:
                part_path.unlink(missing_ok=True)
                raise ValueError(f'文件超过预期大小: {current_pos}/{total_size}')

            stalled_retries = 0
            now = time.time()
            if now - last_log >= 5 or (total_size and current_pos >= total_size):
                if total_size:
                    percent = min(current_pos / total_size * 100, 100)
                    print(f'\r[进度] {video_prefix}{percent:.1f}% ({current_pos}/{total_size})', end='', flush=True)
                else:
                    print(f'\r[进度] {video_prefix}已下载 {current_pos} 字节', end='', flush=True)
                last_log = now

        except (RequestException, ChunkedEncodingError, ValueError, OSError) as e:
            current_pos = part_path.stat().st_size if part_path.exists() else 0
            made_progress = current_pos > start_pos
            stalled_retries = 0 if made_progress else stalled_retries + 1
            segment_size = MIN_SEGMENT_SIZE if made_progress else segment_size

            _merge_retry_info(retry_info_list, retry_lock, {
                'video_info': video_info or str(filepath.name),
                'file_size_mb': total_size / 1024 / 1024 if total_size else 0,
                'retry_count': 1,
                'error_type': type(e).__name__,
                'downloaded_mb': current_pos / 1024 / 1024,
                'remaining_mb': (total_size - current_pos) / 1024 / 1024 if total_size and current_pos else 0,
                'error_msg': str(e)[:100],
                '_increment_retry': True,
            })

            if stalled_retries >= max_retries:
                _mark_retry_status(retry_info_list, retry_lock, video_info or str(filepath.name), '失败')
                print(f'\n[失败] {video_prefix}连续{max_retries}次分段无进展，保留断点文件: {part_path}')
                return _failure(filepath)

            wait_time = 1 if made_progress else min(2 + stalled_retries, 10)
            progress_text = '本段有进展' if made_progress else f'连续无进展{stalled_retries}次'
            print(f'\n[分段重试] {video_prefix}{type(e).__name__}: {e}；{progress_text}，{wait_time}秒后继续')
            time.sleep(wait_time)

    actual_size = part_path.stat().st_size if part_path.exists() else 0
    if total_size and actual_size != total_size:
        _mark_retry_status(retry_info_list, retry_lock, video_info or str(filepath.name), '失败')
        print(f'\n[失败] {video_prefix}文件大小不匹配: {actual_size}/{total_size}')
        return _failure(filepath)
    if not verify_mp4_file(part_path):
        part_path.unlink(missing_ok=True)
        _mark_retry_status(retry_info_list, retry_lock, video_info or str(filepath.name), '失败')
        print(f'\n[失败] {video_prefix}MP4 文件头校验失败')
        return _failure(filepath)

    if not _replace_with_retry(part_path, filepath, video_prefix):
        _mark_retry_status(retry_info_list, retry_lock, video_info or str(filepath.name), '失败')
        return _failure(filepath)

    _mark_retry_status(retry_info_list, retry_lock, video_info or str(filepath.name), '已完成')
    print(f'\n[完成] {video_prefix}下载成功: {filepath.name}')
    return DownloadOutcome(success=True, downloaded=True, filepath=filepath)
