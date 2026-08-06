import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .utils import verify_mp4_file


HARDWARE_ENCODERS = (
    ('h264_nvenc', ('-c:v', 'h264_nvenc', '-preset', 'fast', '-cq', '21', '-b:v', '0')),
    ('h264_qsv', ('-c:v', 'h264_qsv', '-global_quality', '23')),
    ('h264_amf', ('-c:v', 'h264_amf', '-quality', 'speed', '-qp_i', '22', '-qp_p', '22')),
)
CPU_ENCODER = ('libx264', ('-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20'))
_ENCODER_CACHE = {}


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


def _available_encoders(ffmpeg_path):
    result = subprocess.run(
        [ffmpeg_path, '-hide_banner', '-encoders'],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return set()

    encoders = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            encoders.add(parts[1])
    return encoders


def _encoder_attempts(ffmpeg_path):
    cached_encoder = _ENCODER_CACHE.get(ffmpeg_path)
    if cached_encoder:
        if cached_encoder[0] == CPU_ENCODER[0]:
            return [cached_encoder]
        return [cached_encoder, CPU_ENCODER]

    available = _available_encoders(ffmpeg_path)
    hardware_attempts = [encoder for encoder in HARDWARE_ENCODERS if encoder[0] in available]
    return [*hardware_attempts, CPU_ENCODER]


def _output_path(input_path, upscale_config, output_height=None):
    height = output_height or upscale_config.target_height
    return Path(upscale_config.output_dir) / f'{input_path.stem}_{height}p.mp4'


def _temp_output_path(output_path):
    return output_path.with_name(f'{output_path.stem}.tmp{output_path.suffix}')


def _video_only_temp_output_path(output_path):
    return output_path.with_name(f'{output_path.stem}.video_tmp{output_path.suffix}')


def _seek_args(seconds):
    return ['-ss', str(seconds)] if seconds > 0 else []


def _duration_args(seconds):
    return ['-t', str(seconds)] if seconds > 0 else []


def _video_filter(upscale_config):
    return f'scale=-2:{upscale_config.target_height}:flags=bicubic,unsharp=3:3:0.35,format=yuv420p'


def _run_ffmpeg_encode(input_path, output_path, upscale_config, tool_config, video_filter=None):
    start_time = time.time()
    temp_output_path = _temp_output_path(output_path)
    last_error = None

    for encoder_name, encoder_args in _encoder_attempts(tool_config.ffmpeg_path):
        temp_output_path.unlink(missing_ok=True)
        try:
            filter_args = ['-vf', video_filter] if video_filter else ['-vf', 'format=yuv420p']
            _run([
                tool_config.ffmpeg_path,
                '-hide_banner',
                '-loglevel', 'error',
                '-y',
                *_seek_args(upscale_config.trim_head_seconds),
                *_duration_args(upscale_config.trim_duration_seconds),
                '-i', str(input_path),
                *filter_args,
                *encoder_args,
                '-c:a', 'copy',
                str(temp_output_path),
            ])
            if not verify_mp4_file(temp_output_path):
                raise RuntimeError(f'FFmpeg 输出文件校验失败: {temp_output_path}')

            _ENCODER_CACHE[tool_config.ffmpeg_path] = (encoder_name, encoder_args)
            temp_output_path.replace(output_path)
            return encoder_name, time.time() - start_time
        except Exception as e:
            last_error = e
            temp_output_path.unlink(missing_ok=True)

    raise RuntimeError(f'FFmpeg 处理失败: {last_error}')


def _enhance_with_ffmpeg(input_path, output_path, upscale_config, tool_config):
    if not _tool_exists(tool_config.ffmpeg_path):
        return UpscaleOutcome(False, True, None, f'未找到 FFmpeg: {tool_config.ffmpeg_path}')

    encoder_name, elapsed = _run_ffmpeg_encode(
        input_path,
        output_path,
        upscale_config,
        tool_config,
        video_filter=_video_filter(upscale_config),
    )
    return UpscaleOutcome(True, False, output_path, f'FFmpeg 增强完成({encoder_name}): {output_path}', elapsed)


def _trim_with_ffmpeg(input_path, output_path, upscale_config, tool_config):
    """裁掉片头片尾：优先使用 -c copy 流拷贝（快、无损、不膨胀）；失败时回退到重编码。"""
    if not _tool_exists(tool_config.ffmpeg_path):
        return UpscaleOutcome(False, True, None, f'未找到 FFmpeg: {tool_config.ffmpeg_path}')

    start_time = time.time()
    temp_output_path = _temp_output_path(output_path)
    temp_output_path.unlink(missing_ok=True)
    try:
        _run([
            tool_config.ffmpeg_path,
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            *_seek_args(upscale_config.trim_head_seconds),
            *_duration_args(upscale_config.trim_duration_seconds),
            '-i', str(input_path),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            str(temp_output_path),
        ])
        if not verify_mp4_file(temp_output_path):
            raise RuntimeError(f'FFmpeg 流拷贝输出校验失败: {temp_output_path}')
        temp_output_path.replace(output_path)
        elapsed = time.time() - start_time
        return UpscaleOutcome(True, False, output_path, f'FFmpeg 裁剪完成(流拷贝): {output_path}', elapsed)
    except Exception:
        temp_output_path.unlink(missing_ok=True)
        # 流拷贝失败（如时间戳异常）时回退到重编码
        encoder_name, elapsed = _run_ffmpeg_encode(input_path, output_path, upscale_config, tool_config)
        return UpscaleOutcome(True, False, output_path, f'FFmpeg 裁剪完成(重编码/{encoder_name}): {output_path}', elapsed)


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


def _attach_cover(video_path, output_path, cover_path, tool_config):
    if not cover_path:
        return Path(video_path)

    video_path = Path(video_path)
    output_path = Path(output_path)
    cover_path = Path(cover_path)
    if not cover_path.exists():
        return video_path

    normalized_cover_path = output_path.with_name(f'{output_path.stem}.cover_tmp.jpg')
    temp_output_path = _temp_output_path(output_path)
    normalized_cover_path.unlink(missing_ok=True)
    temp_output_path.unlink(missing_ok=True)
    _run([
        tool_config.ffmpeg_path,
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        '-i', str(cover_path),
        '-frames:v', '1',
        str(normalized_cover_path),
    ])
    _run([
        tool_config.ffmpeg_path,
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        '-i', str(video_path),
        '-i', str(normalized_cover_path),
        '-map', '0',
        '-map', '1',
        '-c', 'copy',
        '-disposition:v:1', 'attached_pic',
        str(temp_output_path),
    ])
    if not verify_mp4_file(temp_output_path):
        temp_output_path.unlink(missing_ok=True)
        normalized_cover_path.unlink(missing_ok=True)
        raise RuntimeError(f'封面写入后文件校验失败: {temp_output_path}')
    temp_output_path.replace(output_path)
    normalized_cover_path.unlink(missing_ok=True)
    return output_path


def _finish_processed_video(input_path, output_path, upscale_config, tool_config, cover_path, processor):
    input_path = Path(input_path)
    output_dir = Path(upscale_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    working_output_path = _video_only_temp_output_path(output_path) if cover_path else output_path

    if output_path.exists() and verify_mp4_file(output_path):
        return UpscaleOutcome(True, True, output_path, f'输出已存在: {output_path}')
    if output_path.exists():
        output_path.unlink()
    if working_output_path.exists():
        working_output_path.unlink()
    result = processor(input_path, working_output_path, upscale_config, tool_config)

    final_path = _attach_cover(working_output_path, output_path, cover_path, tool_config)
    if final_path != working_output_path and working_output_path.exists():
        working_output_path.unlink()
    return UpscaleOutcome(result.success, result.skipped, output_path, result.message.replace(str(working_output_path), str(output_path)), result.elapsed_seconds)


def enhance_video_to_1080p(input_path, upscale_config, tool_config, cover_path=None):
    """按配置生成增强版视频。"""
    input_path = Path(input_path)
    output_path = _output_path(input_path, upscale_config)
    processor = _enhance_with_realesrgan if upscale_config.enhance_engine == 'realesrgan' else _enhance_with_ffmpeg
    return _finish_processed_video(input_path, output_path, upscale_config, tool_config, cover_path, processor)


def trim_video(input_path, upscale_config, tool_config, cover_path=None, source_height=0):
    """只裁剪片头片尾，保留原分辨率。"""
    input_path = Path(input_path)
    output_path = _output_path(input_path, upscale_config, output_height=source_height or upscale_config.target_height)
    return _finish_processed_video(input_path, output_path, upscale_config, tool_config, cover_path, _trim_with_ffmpeg)
