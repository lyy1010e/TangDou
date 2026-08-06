from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadConfig:
    output_dir: Path = Path('download')
    debug_api_response: bool = False


@dataclass(frozen=True)
class UpscaleConfig:
    # 增强开关：true 时对低分辨率源做超分增强；false 时只下载/裁剪不增强。
    enabled: bool = True
    # 去片头片尾开关：true 时根据视频的 head_t/end_t 标记裁掉片头片尾；与增强开关相互独立。
    trim_enabled: bool = True
    enhance_engine: str = 'ffmpeg'
    target_height: int = 720
    trim_head_seconds: float = 0.0
    trim_duration_seconds: float = 0.0
    output_dir: Path = Path('download/enhanced')
    temp_dir: Path = Path('E:/tangdou_upscale_temp')


@dataclass(frozen=True)
class ToolConfig:
    ffmpeg_path: str = 'ffmpeg'
    realesrgan_path: str = 'realesrgan-ncnn-vulkan'


@dataclass(frozen=True)
class AppConfig:
    download: DownloadConfig
    upscale: UpscaleConfig
    tools: ToolConfig


DEFAULT_CONFIG = AppConfig(
    download=DownloadConfig(),
    upscale=UpscaleConfig(),
    tools=ToolConfig(),
)


def _get_bool(parser, section, option, default):
    try:
        return parser.getboolean(section, option, fallback=default)
    except ValueError:
        print(f'[警告] config.ini 中 {section}.{option} 不是有效布尔值，已使用默认值: {default}')
        return default


def _get_int(parser, section, option, default, min_value=None):
    try:
        value = parser.getint(section, option, fallback=default)
    except ValueError:
        print(f'[警告] config.ini 中 {section}.{option} 不是有效整数，已使用默认值: {default}')
        return default

    if min_value is not None and value < min_value:
        print(f'[警告] config.ini 中 {section}.{option} 小于 {min_value}，已使用默认值: {default}')
        return default
    return value


def _get_float(parser, section, option, default, min_value=None):
    try:
        value = parser.getfloat(section, option, fallback=default)
    except ValueError:
        print(f'[警告] config.ini 中 {section}.{option} 不是有效数字，已使用默认值: {default}')
        return default

    if min_value is not None and value < min_value:
        print(f'[警告] config.ini 中 {section}.{option} 小于 {min_value}，已使用默认值: {default}')
        return default
    return value


def _get_choice(parser, section, option, default, choices):
    value = parser.get(section, option, fallback=default).strip().lower()
    if value in choices:
        return value
    print(f'[警告] config.ini 中 {section}.{option} 只能是 {", ".join(sorted(choices))}，已使用默认值: {default}')
    return default


def load_config(config_path='config.ini'):
    """读取 config.ini，缺失或配置错误时回退到默认值。"""
    path = Path(config_path)
    parser = ConfigParser()
    defaults = DEFAULT_CONFIG

    if not path.exists():
        print('[提示] 未找到 config.ini，已使用默认配置')
        return defaults

    parser.read(path, encoding='utf-8')

    download = DownloadConfig(
        output_dir=Path(parser.get('download', 'output_dir', fallback=str(defaults.download.output_dir))),
        debug_api_response=_get_bool(parser, 'download', 'debug_api_response', defaults.download.debug_api_response),
    )
    upscale = UpscaleConfig(
        enabled=_get_bool(parser, 'upscale', 'enabled', defaults.upscale.enabled),
        trim_enabled=_get_bool(parser, 'upscale', 'trim_enabled', defaults.upscale.trim_enabled),
        enhance_engine=_get_choice(parser, 'upscale', 'enhance_engine', defaults.upscale.enhance_engine, {'ffmpeg', 'realesrgan'}),
        target_height=_get_int(parser, 'upscale', 'target_height', defaults.upscale.target_height, min_value=1),
        output_dir=Path(parser.get('upscale', 'output_dir', fallback=str(defaults.upscale.output_dir))),
        temp_dir=Path(parser.get(
            'realesrgan',
            'temp_dir',
            fallback=parser.get('upscale', 'temp_dir', fallback=str(defaults.upscale.temp_dir)),
        )),
    )
    tools = ToolConfig(
        ffmpeg_path=parser.get(
            'ffmpeg',
            'path',
            fallback=parser.get('tools', 'ffmpeg_path', fallback=defaults.tools.ffmpeg_path),
        ),
        realesrgan_path=parser.get(
            'realesrgan',
            'path',
            fallback=parser.get('tools', 'realesrgan_path', fallback=defaults.tools.realesrgan_path),
        ),
    )
    return AppConfig(download=download, upscale=upscale, tools=tools)
