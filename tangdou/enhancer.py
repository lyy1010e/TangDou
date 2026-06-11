import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .utils import verify_mp4_file


@dataclass(frozen=True)
class UpscaleOutcome:
    success: bool
    skipped: bool
    output_path: Path | None
    message: str
    elapsed_seconds: float = 0.0


def _tool_exists(command):
    path = Path(command)
    return path.exists() if path.parent != Path('.') else shutil.which(command) is not None


def _run(command):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        error_text = (result.stderr or result.stdout or '').strip()
        raise RuntimeError(error_text or f'命令执行失败: {command[0]}')


def _output_path(input_path, upscale_config):
    return Path(upscale_config.output_dir) / f'{input_path.stem}_{upscale_config.target_height}p.mp4'


def _temp_output_path(output_path):
    return output_path.with_name(f'{output_path.stem}.tmp{output_path.suffix}')


def _seek_args(seconds):
    return ['-ss', str(seconds)] if seconds > 0 else []


def _duration_args(seconds):
    return ['-t', str(seconds)] if seconds > 0 else []


def _video_filter(upscale_config):
    return f'scale=-2:{upscale_config.target_height}:flags=bicubic,unsharp=3:3:0.6'


def _enhance_with_ffmpeg(input_path, output_path, upscale_config, tool_config):
    if not _tool_exists(tool_config.ffmpeg_path):
        return UpscaleOutcome(False, True, None, f'未找到 FFmpeg: {tool_config.ffmpeg_path}')

    start_time = time.time()
    temp_output_path = _temp_output_path(output_path)
    temp_output_path.unlink(missing_ok=True)
    _run([
        tool_config.ffmpeg_path,
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        *_seek_args(upscale_config.trim_head_seconds),
        *_duration_args(upscale_config.trim_duration_seconds),
        '-i', str(input_path),
        '-vf', _video_filter(upscale_config),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '20',
        '-c:a', 'copy',
        str(temp_output_path),
    ])
    if not verify_mp4_file(temp_output_path):
        temp_output_path.unlink(missing_ok=True)
        raise RuntimeError(f'FFmpeg 输出文件校验失败: {temp_output_path}')
    temp_output_path.replace(output_path)
    elapsed = time.time() - start_time
    return UpscaleOutcome(True, False, output_path, f'FFmpeg 增强完成: {output_path}', elapsed)


def _enhance_with_realesrgan(input_path, output_path, upscale_config, tool_config):
    if not _tool_exists(tool_config.ffmpeg_path):
        return UpscaleOutcome(False, True, None, f'未找到 FFmpeg: {tool_config.ffmpeg_path}')
    if not _tool_exists(tool_config.realesrgan_path):
        return UpscaleOutcome(False, True, None, f'未找到 Real-ESRGAN: {tool_config.realesrgan_path}')

    start_time = time.time()
    temp_root = Path(upscale_config.temp_dir)
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_output_path = _temp_output_path(output_path)
    temp_output_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix='tangdou_upscale_', dir=temp_root) as temp_dir:
        temp_path = Path(temp_dir)
        frames_dir = temp_path / 'frames'
        upscaled_dir = temp_path / 'upscaled'
        frames_dir.mkdir()
        upscaled_dir.mkdir()

        frame_pattern = str(frames_dir / 'frame_%08d.png')
        upscaled_pattern = str(upscaled_dir / 'frame_%08d.png')

        _run([
            tool_config.ffmpeg_path,
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            *_seek_args(upscale_config.trim_head_seconds),
            *_duration_args(upscale_config.trim_duration_seconds),
            '-i', str(input_path),
            frame_pattern,
        ])
        _run([
            tool_config.realesrgan_path,
            '-i', str(frames_dir),
            '-o', str(upscaled_dir),
            '-n', 'realesr-animevideov3',
            '-s', '2',
            '-f', 'png',
        ])
        _run([
            tool_config.ffmpeg_path,
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-framerate', '25',
            '-i', upscaled_pattern,
            *_seek_args(upscale_config.trim_head_seconds),
            *_duration_args(upscale_config.trim_duration_seconds),
            '-i', str(input_path),
            '-map', '0:v:0',
            '-map', '1:a?',
            '-vf', f'scale=-2:{upscale_config.target_height}',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '20',
            '-c:a', 'copy',
            '-shortest',
            str(temp_output_path),
        ])

    if not verify_mp4_file(temp_output_path):
        temp_output_path.unlink(missing_ok=True)
        raise RuntimeError(f'Real-ESRGAN 输出文件校验失败: {temp_output_path}')
    temp_output_path.replace(output_path)
    elapsed = time.time() - start_time
    return UpscaleOutcome(True, False, output_path, f'Real-ESRGAN 增强完成: {output_path}', elapsed)


def enhance_video_to_1080p(input_path, upscale_config, tool_config):
    """按配置生成增强版视频。"""
    input_path = Path(input_path)
    output_dir = Path(upscale_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(input_path, upscale_config)

    if output_path.exists() and verify_mp4_file(output_path):
        return UpscaleOutcome(True, True, output_path, f'增强版已存在: {output_path}')
    if output_path.exists():
        output_path.unlink()
    if not upscale_config.enabled:
        return UpscaleOutcome(True, True, None, '视频增强未启用')
    if upscale_config.enhance_engine == 'realesrgan':
        return _enhance_with_realesrgan(input_path, output_path, upscale_config, tool_config)
    return _enhance_with_ffmpeg(input_path, output_path, upscale_config, tool_config)
