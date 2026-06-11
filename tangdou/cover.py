from pathlib import Path
from urllib.parse import urljoin

import requests


TANGDOU_BASE_URL = 'https://www.tangdou.com'


def resolve_cover_url(data, play_data=None):
    """从列表接口或播放接口中解析封面 URL。"""
    cover = data.get('pic') or data.get('cover')
    if not cover and isinstance(play_data, dict):
        play_body = play_data.get('data', {})
        if isinstance(play_body, dict):
            cover = play_body.get('cover') or play_body.get('pic')
    if not cover:
        return ''
    return cover if cover.startswith(('http://', 'https://')) else urljoin(TANGDOU_BASE_URL, cover)


def download_cover(cover_url, output_path, headers):
    """下载封面图，失败时返回 None，不影响视频下载和增强。"""
    if not cover_url:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(cover_url, headers=headers, timeout=(10, 30))
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'image' not in content_type:
            return None
        output_path.write_bytes(response.content)
        return output_path
    except Exception as e:
        print(f'[警告] 封面下载失败: {e}')
        return None
