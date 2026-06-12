from hashlib import md5
from urllib.parse import parse_qsl, urlencode, unquote, urlparse, urlunparse


TANGDOU_SIGN_SALT = '305%daf5g7ra05$#+6%pm!ud922u!(_t#elidt7q2t'


def _query_without_hash(query):
    parts = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key != 'hash':
            parts.append((key, value))
    return urlencode(parts)


def sign_query(query):
    """按糖豆 APP 规则重新计算 query 的 hash。"""
    unsigned_query = _query_without_hash(query)
    return md5((unquote(unsigned_query) + TANGDOU_SIGN_SALT).encode()).hexdigest()


def build_page_url(template_url, page):
    """基于一条收藏页 URL 模板生成指定页 URL。"""
    normalized_url = template_url.replace('&amp;', '&')
    parsed = urlparse(normalized_url)
    params = []
    has_page = False
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key == 'hash':
            continue
        if key == 'page':
            value = str(page)
            has_page = True
        params.append((key, value))

    if not has_page:
        params.append(('page', str(page)))

    query_without_hash = urlencode(params)
    query = f'{query_without_hash}&hash={sign_query(query_without_hash)}'
    return urlunparse(parsed._replace(query=query))
