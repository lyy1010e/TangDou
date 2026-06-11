import re
from dataclasses import dataclass


VIDEO_URL_KEYS = ('url', 'play', 'video', 'src', 'm3u8', 'mp4')
HIGH_QUALITY_WORDS = ('hd', 'uhd', 'fhd', '1080', '720', '超清', '高清', '蓝光', '原画')
LOW_QUALITY_WORDS = ('sd', 'ld', '360', '480', '标清', '流畅')


@dataclass(frozen=True)
class SourceChoice:
    url: str
    label: str
    is_hd: bool
    score: int


def _is_video_url(value, key_path):
    text = str(value)
    key_text = '.'.join(key_path).lower()
    looks_like_url = text.startswith(('http://', 'https://'))
    key_matches = any(word in key_text for word in VIDEO_URL_KEYS)
    value_matches = re.search(r'\.(mp4|m3u8)(\?|$)', text.lower()) is not None
    return looks_like_url and (key_matches or value_matches)


def _to_int(value):
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _height_from_text(text):
    match = re.search(r'(?<!\d)(2160|1440|1080|720|540|480|360)p?(?!\d)', text.lower())
    return int(match.group(1)) if match else 0


def source_height(choice):
    """从候选源标签和 URL 中识别视频高度，识别不到返回 0。"""
    return _height_from_text(f'{choice.label} {choice.url}')


def _score(node, key_path):
    label = '.'.join(key_path)
    text = label.lower()
    height = 0
    bitrate = 0

    if isinstance(node, dict):
        height = max(_to_int(node.get('height')), _height_from_text(str(node)))
        width = _to_int(node.get('width'))
        bitrate = max(_to_int(node.get('bitrate')), _to_int(node.get('vb')), _to_int(node.get('rate')))
        if not height and width:
            height = int(width * 9 / 16)
        text = f'{text} {node}'.lower()
    else:
        height = _height_from_text(f'{label} {node}')

    quality_bonus = 0
    if any(word in text for word in HIGH_QUALITY_WORDS):
        quality_bonus += 500_000
    if any(word in text for word in LOW_QUALITY_WORDS):
        quality_bonus -= 200_000
    return height * 1_000 + bitrate + quality_bonus


def _collect_candidates(node, key_path=()):
    candidates = []
    if isinstance(node, dict):
        video_url_count = sum(
            1 for key, value in node.items()
            if isinstance(value, str) and _is_video_url(value, (*key_path, str(key)))
        )
        for key, value in node.items():
            child_path = (*key_path, str(key))
            if isinstance(value, str) and _is_video_url(value, child_path):
                own_score = _score({str(key): value}, child_path)
                score = max(own_score, _score(node, child_path)) if video_url_count == 1 else own_score
                candidates.append(SourceChoice(
                    url=value,
                    label='.'.join(child_path),
                    is_hd=False,
                    score=score,
                ))
            candidates.extend(_collect_candidates(value, child_path))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            candidates.extend(_collect_candidates(item, (*key_path, str(index))))
    return candidates


def _with_hd_flag(choice, fallback):
    label_text = choice.label.lower()
    is_high_label = any(word in label_text for word in HIGH_QUALITY_WORDS)
    is_better_than_fallback = choice.url != fallback.url and choice.score > fallback.score
    return SourceChoice(
        url=choice.url,
        label=choice.label,
        is_hd=is_high_label or is_better_than_fallback,
        score=choice.score,
    )


def select_best_source(play_data, prefer_highest_source=True):
    """从播放接口数据中选择最高质量视频源，异常结构回退到 play_url。"""
    data = play_data.get('data', {}) if isinstance(play_data, dict) else {}
    fallback_url = data.get('play_url') if isinstance(data, dict) else None
    if not fallback_url:
        return SourceChoice(url='', label='未找到播放地址', is_hd=False, score=0)

    fallback = SourceChoice(
        url=fallback_url,
        label='data.play_url',
        is_hd=False,
        score=_score({'play_url': fallback_url}, ('data', 'play_url')),
    )
    candidates = _collect_candidates(data, ('data',))
    candidates.append(fallback)

    if not prefer_highest_source:
        return fallback

    best = max(candidates, key=lambda item: item.score)
    return _with_hd_flag(best, fallback)


def collect_source_candidates(play_data):
    """返回播放接口里识别到的视频 URL 候选，用于调试源选择。"""
    data = play_data.get('data', {}) if isinstance(play_data, dict) else {}
    if not isinstance(data, dict):
        return []
    return sorted(_collect_candidates(data, ('data',)), key=lambda item: item.score, reverse=True)
