import re
import threading
from pathlib import Path


_print_lock = threading.Lock()


def thread_safe_print(*args, **kwargs):
    """线程安全的打印函数，防止多线程日志混乱。"""
    with _print_lock:
        print(*args, **kwargs)


def clean_filename(filename):
    """清理文件名，移除 Windows 文件名非法字符。"""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename[:200] if len(filename) > 200 else filename


def verify_mp4_file(filepath):
    """验证 MP4 文件头是否符合常见 MP4/MOV 格式。"""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return False

        file_size = filepath.stat().st_size
        if file_size < 8:
            return False

        with open(filepath, 'rb') as f:
            header = f.read(min(32, file_size))

        if len(header) < 8:
            return False
        if header[4:8] == b'ftyp' or b'ftyp' in header:
            return True

        mp4_indicators = [b'moov', b'mdat', b'mvhd', b'trak', b'mdhd']
        return file_size > 1024 and any(indicator in header for indicator in mp4_indicators)
    except Exception as e:
        print(f'[警告] 文件验证出错: {e}')
        return False
