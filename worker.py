import base64
import hashlib
import html
import os
import re
import time
from typing import Any, Dict, List, Tuple

import feedparser
import httpx
from html.parser import HTMLParser

from app import (
    CFG,
    DEEPL_API_KEY,
    DEEPL_ENDPOINT,
    LIBRETRANSLATE_API_KEY,
    LIBRETRANSLATE_ENDPOINT,
    cache_get,
    cache_key_for_item,
    cache_put,
    cache_purge_old,
    build_translated_feed_xml,
    clamp,
    entry_text,
    feed_cache_put,
    pick_item_id,
    protect_iocs,
    protect_breaks,
    protect_markers,
    restore_iocs,
    restore_breaks,
    restore_markers,
    text_hash,
)


class _DeepLQuotaExceeded(RuntimeError):
    pass


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}", flush=True)


_LOG_ONCE: set[str] = set()


def log_once(msg: str) -> None:
    if msg in _LOG_ONCE:
        return
    _LOG_ONCE.add(msg)
    log(msg)


class _TextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "p",
        "div",
        "br",
        "pre",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._in_pre = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        if tag == "pre":
            self._in_pre = True
            self._newline()
            self._parts.append("[[[PRE]]]")
            self._newline()
        elif tag in self._BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self._in_pre = False
            self._newline()
            self._parts.append("[[[/PRE]]]")
            self._newline()
        elif tag in self._BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if not data or data.isspace():
            return
        if self._in_pre:
            self._parts.append(data)
        else:
            self._parts.append(re.sub(r"\s+", " ", data.strip()))

    def _newline(self) -> None:
        if not self._parts:
            return
        if self._parts[-1] != "\n":
            self._parts.append("\n")

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    # Strip script/style blocks first
    html = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", html)
    # Preserve images as markers; store data URIs to disk and emit URLs.
    def _img_repl(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith("data:image/"):
            name = store_data_image(src)
            if name:
                return f"\n[[[IMGURL:{name}]]]\n"
        return f"\n[[[IMG:{src}]]]\n"

    html = re.sub(
        r'(?is)<img[^>]*?src=["\']([^"\']+)["\'][^>]*>',
        _img_repl,
        html,
    )
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def store_data_image(data_uri: str) -> str:
    m = re.match(r"^data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=]+)$", data_uri)
    if not m:
        return ""
    mime = m.group(1).lower()
    b64 = m.group(2)
    try:
        data = base64.b64decode(b64)
    except Exception:
        return ""
    ext = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
    }.get(mime, ".bin")
    digest = hashlib.sha256(data).hexdigest()
    name = f"{digest}{ext}"
    os.makedirs(CFG.image_dir, exist_ok=True)
    path = os.path.join(CFG.image_dir, name)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return name


def extract_sections(text: str, headings: List[str]) -> str:
    if not text or not headings:
        return text or ""
    want = {h.strip() for h in headings if h and h.strip()}
    if not want:
        return text

    lines = [ln.strip() for ln in text.splitlines()]
    sections: List[str] = []
    cur_heading = None
    cur_lines: List[str] = []

    def flush() -> None:
        if cur_heading and cur_lines:
            sections.append(cur_heading)
            sections.append("\n".join(cur_lines).strip())

    for ln in lines:
        if not ln:
            if cur_lines:
                cur_lines.append("")
            continue
        if ln in want:
            flush()
            cur_heading = ln
            cur_lines = []
            continue
        if cur_heading:
            cur_lines.append(ln)

    flush()
    return "\n\n".join([s for s in sections if s]).strip() or text


def fetch_full_text(url: str) -> str:
    if not url:
        return ""
    headers = {"User-Agent": "rss-translate-proxy/1.0"}
    with httpx.Client(timeout=CFG.full_content_timeout_seconds) as client:
        r = client.get(url, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return html_to_text(r.text)


def extract_article_id(entry: Any) -> str:
    link = getattr(entry, "link", None) or entry.get("link") or ""
    guid = getattr(entry, "id", None) or entry.get("guid") or ""
    for candidate in (link, guid):
        if not candidate:
            continue
        m = re.search(r"(\d{5,})", str(candidate))
        if m:
            return m.group(1)
    return ""


def json_path_get(data: Any, path: str) -> Any:
    if not path:
        return None
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def fetch_full_text_via_api(feed_cfg, entry: Any) -> str:
    if not feed_cfg.full_content_api_url_template:
        return ""
    article_id = extract_article_id(entry)
    if not article_id:
        return ""
    url = feed_cfg.full_content_api_url_template.replace("{id}", article_id)
    headers = {"User-Agent": "rss-translate-proxy/1.0"}
    with httpx.Client(timeout=CFG.full_content_timeout_seconds) as client:
        r = client.get(url, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        text = ""
        if feed_cfg.full_content_api_format == "xml":
            # Extract <tag>...</tag> content and unescape HTML entities.
            tags = [t.strip() for t in (feed_cfg.full_content_api_text_path or "text").split(",")]
            parts = []
            for tag in tags:
                if not tag:
                    continue
                m = re.search(
                    rf"<{tag}>(.*?)</{tag}>",
                    r.text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if m:
                    parts.append(html.unescape(m.group(1)))
            if not parts:
                return ""
            text = "\n\n".join(parts)
        else:
            data = r.json()
            paths = [p.strip() for p in (feed_cfg.full_content_api_text_path or "").split(",")]
            parts = []
            for path in paths:
                if not path:
                    continue
                val = json_path_get(data, path)
                if val:
                    parts.append(str(val))
            if not parts:
                return ""
            text = "\n\n".join(parts)
        if feed_cfg.full_content_is_html:
            text = html_to_text(str(text))
        text = extract_sections(text, feed_cfg.full_content_extract_sections or [])
        return str(text)


def deepl_translate_sync(texts: List[str], target_lang: str) -> List[str]:
    if not DEEPL_API_KEY:
        raise RuntimeError("DEEPL_API_KEY not set")

    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
    payload = {"text": texts, "target_lang": target_lang}

    with httpx.Client(timeout=30) as client:
        r = client.post(DEEPL_ENDPOINT, json=payload, headers=headers)
        if r.status_code != 200:
            msg = f"DeepL error {r.status_code}: {r.text}"
            if r.status_code == 456 or "Quota exceeded" in r.text:
                raise _DeepLQuotaExceeded(msg)
            raise RuntimeError(msg)

        payload = r.json()
        translations = payload.get("translations", [])
        return [tr.get("text", "") for tr in translations]


def _lt_endpoint() -> str:
    ep = (LIBRETRANSLATE_ENDPOINT or "").strip()
    if not ep:
        raise RuntimeError("LIBRETRANSLATE_ENDPOINT not set")
    if ep.endswith("/translate"):
        return ep
    return ep.rstrip("/") + "/translate"


def _lt_ready(timeout_seconds: int = 180) -> bool:
    base = _lt_endpoint().rsplit("/translate", 1)[0]
    url = base + "/languages"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5) as client:
                r = client.get(url)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def libretranslate_sync(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    payload = {
        "q": texts,
        "source": source_lang or "auto",
        "target": target_lang,
        "format": "text",
    }
    if LIBRETRANSLATE_API_KEY:
        payload["api_key"] = LIBRETRANSLATE_API_KEY

    with httpx.Client(timeout=180) as client:
        r = client.post(_lt_endpoint(), json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"LibreTranslate error {r.status_code}: {r.text}")

        data = r.json()
        translations = data.get("translatedText")
        if isinstance(translations, list):
            return [t or "" for t in translations]
        # Some servers return a single string for single input
        if isinstance(translations, str):
            return [translations]
        raise RuntimeError("LibreTranslate response missing translatedText")


def translate_sync(texts: List[str], target_lang: str) -> List[str]:
    provider = (CFG.translation_provider or "deepl").lower()
    if provider == "deepl":
        try:
            return deepl_translate_sync(texts, target_lang)
        except _DeepLQuotaExceeded:
            log_once("deepl quota exceeded; falling back to libretranslate")
        except RuntimeError as e:
            if "DEEPL_API_KEY not set" in str(e):
                log_once("deepl key missing; falling back to libretranslate")
            else:
                raise
    if provider == "libretranslate":
        # LibreTranslate expects lower-case language codes like 'uk' and 'en'
        if not _lt_ready():
            raise RuntimeError("LibreTranslate not ready (languages endpoint timeout)")
        lt_target = target_lang.lower()
        lt_source = (CFG.source_lang or "auto").lower()
        return _translate_with_chunking(texts, lt_source, lt_target)
    # Fallback to LibreTranslate when DeepL is unavailable or quota exceeded
    log_once("deepl unavailable; using libretranslate")
    if not _lt_ready():
        raise RuntimeError("LibreTranslate not ready (languages endpoint timeout)")
    lt_target = target_lang.lower()
    lt_source = (CFG.source_lang or "auto").lower()
    return _translate_with_chunking(texts, lt_source, lt_target)


def _chunk_text(text: str, limit: int = 4000) -> List[str]:
    if not text:
        return [""]
    parts: List[str] = []
    buf: List[str] = []
    size = 0
    for para in re.split(r"\n{2,}", text):
        p = para.strip()
        if not p:
            continue
        add = len(p) + 2
        if size + add > limit and buf:
            parts.append("\n\n".join(buf))
            buf = [p]
            size = len(p)
        else:
            buf.append(p)
            size += add
    if buf:
        parts.append("\n\n".join(buf))
    return parts or [text]


def _translate_with_chunking(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    out: List[str] = []
    for t in texts:
        chunks = _chunk_text(t)
        if len(chunks) == 1:
            out.extend(libretranslate_sync([t], source_lang, target_lang))
            continue
        translated_chunks: List[str] = []
        for i, c in enumerate(chunks, start=1):
            log(f"libretranslate chunk {i}/{len(chunks)} len={len(c)}")
            translated_chunks.extend(libretranslate_sync([c], source_lang, target_lang))
        out.append("\n\n".join(translated_chunks))
    return out
    raise RuntimeError(f"Unknown translation provider: {provider}")


def translate_feed(feed_cfg) -> int:
    parsed = feedparser.parse(feed_cfg.source_url)
    if parsed.bozo and not parsed.entries:
        log(f"feed={feed_cfg.id} fetch/parse failed: {parsed.bozo_exception}")
        return 0

    entries = parsed.entries[: feed_cfg.item_limit]

    to_translate: List[
        Tuple[int, str, str, str, Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]
    ] = []
    # (index, item_id, src_hash, cache_key, title_tokens, desc_tokens, title_markers, desc_markers)

    translated_title: Dict[int, str] = {}
    translated_desc: Dict[int, str] = {}
    original_title: Dict[int, str] = {}
    original_desc: Dict[int, str] = {}
    original_link: Dict[int, str] = {}
    clamped_title: Dict[int, str] = {}
    clamped_desc: Dict[int, str] = {}

    for idx, entry in enumerate(entries):
        item_id = pick_item_id(entry)
        title, desc = entry_text(entry)
        link = getattr(entry, "link", None) or entry.get("link") or ""

        if feed_cfg.fetch_full_content and link:
            try:
                if feed_cfg.full_content_api_url_template:
                    full_text = fetch_full_text_via_api(feed_cfg, entry)
                else:
                    full_text = fetch_full_text(str(link))
                if full_text:
                    desc = full_text
            except Exception as e:
                log(f"feed={feed_cfg.id} full-content fetch failed: {e}")

        original_title[idx] = title
        original_desc[idx] = desc
        original_link[idx] = str(link) if link else ""

        title = clamp(title, CFG.max_chars_per_item)
        desc = clamp(desc, CFG.max_chars_per_item)
        clamped_title[idx] = title
        clamped_desc[idx] = desc

        src_h = text_hash(title, desc)
        ckey = cache_key_for_item(feed_cfg.id, item_id, src_h, CFG.target_lang)

        cached = cache_get(CFG.sqlite_path, ckey)
        if cached:
            continue

        if CFG.preserve_iocs:
            title_prot, title_tokens = protect_iocs(title)
            desc_prot, desc_tokens = protect_iocs(desc)
        else:
            title_prot, title_tokens = title, {}
            desc_prot, desc_tokens = desc, {}

        title_prot = protect_breaks(title_prot)
        desc_prot = protect_breaks(desc_prot)

        title_prot, title_markers = protect_markers(title_prot)
        desc_prot, desc_markers = protect_markers(desc_prot)

        to_translate.append(
            (idx, item_id, src_h, ckey, title_tokens, desc_tokens, title_markers, desc_markers)
        )
        translated_title[idx] = title_prot
        translated_desc[idx] = desc_prot

    if to_translate:
        idxs = [t[0] for t in to_translate]
        titles_batch = [translated_title[i] for i in idxs]
        descs_batch = [translated_desc[i] for i in idxs]

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                titles_out = translate_sync(titles_batch, CFG.target_lang)
                descs_out = translate_sync(descs_batch, CFG.target_lang)
                break
            except Exception as e:
                msg = str(e)
                retryable = any(
                    frag in msg
                    for frag in (
                        "LibreTranslate not ready",
                        "timed out",
                        "Connection refused",
                        "connect: connection refused",
                    )
                )
                if retryable and attempt < max_attempts:
                    log(
                        f"feed={feed_cfg.id} translation retry {attempt}/{max_attempts} after error: {msg}"
                    )
                    time.sleep(10)
                    continue
                log(f"feed={feed_cfg.id} translation failed: {e}")
                return 0

        written = 0
        for j, (
            idx,
            item_id,
            src_h,
            ckey,
            title_tokens,
            desc_tokens,
            title_markers,
            desc_markers,
        ) in enumerate(
            to_translate
        ):
            t_title = titles_out[j]
            t_desc = descs_out[j]
            t_title = restore_markers(t_title, title_markers)
            t_desc = restore_markers(t_desc, desc_markers)
            t_title = restore_breaks(t_title)
            t_desc = restore_breaks(t_desc)
            t_title = restore_iocs(t_title, title_tokens)
            t_desc = restore_iocs(t_desc, desc_tokens)

            try:
                cache_put(
                    CFG.sqlite_path,
                    ckey,
                    engine="deepl",
                    target_lang=CFG.target_lang,
                    src_hash=src_h,
                    translated_title=t_title,
                    translated_desc=t_desc,
                )
                written += 1
            except Exception as e:
                log(f"feed={feed_cfg.id} cache write failed: {e}")
    else:
        written = 0

    # Build feed XML from cached translations only and store it
    translated_title = {}
    translated_desc = {}
    for idx, entry in enumerate(entries):
        item_id = pick_item_id(entry)
        title = clamped_title.get(idx, "") or original_title.get(idx, "")
        desc = clamped_desc.get(idx, "") or original_desc.get(idx, "")

        src_h = text_hash(title, desc)
        ckey = cache_key_for_item(feed_cfg.id, item_id, src_h, CFG.target_lang)

        cached = cache_get(CFG.sqlite_path, ckey)
        if not cached:
            continue
        translated_title[idx] = cached.get("translated_title") or ""
        translated_desc[idx] = cached.get("translated_desc") or ""

    xml_bytes = build_translated_feed_xml(
        feed_cfg,
        entries,
        translated_title,
        translated_desc,
        original_title,
        original_desc,
        original_link,
    )
    feed_cache_put(CFG.sqlite_path, feed_cfg.id, xml_bytes.decode("utf-8"))

    return written


def run_once() -> None:
    if not DEEPL_API_KEY:
        log("DEEPL_API_KEY not set; skipping translation run.")
        return

    # Purge old cache opportunistically
    try:
        cache_purge_old(CFG.sqlite_path, CFG.ttl_seconds)
    except Exception as e:
        log(f"cache purge failed: {e}")

    total = 0
    for f in CFG.feeds:
        written = translate_feed(f)
        total += written
        log(f"feed={f.id} translated={written}")

    log(f"run complete translated_total={total}")


def main() -> None:
    log(f"worker started poll_seconds={CFG.poll_seconds}")
    while True:
        run_once()
        time.sleep(max(1, CFG.poll_seconds))


if __name__ == "__main__":
    main()
