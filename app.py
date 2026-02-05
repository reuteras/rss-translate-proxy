import hashlib
import html
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import mimetypes
import yaml
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import FileResponse
from feedgen.feed import FeedGenerator

DEEPL_ENDPOINT = os.environ.get(
    "DEEPL_ENDPOINT", "https://api-free.deepl.com/v2/translate"
)
LIBRETRANSLATE_ENDPOINT = os.environ.get(
    "LIBRETRANSLATE_ENDPOINT", "https://libretranslate.com/translate"
)
LIBRETRANSLATE_API_KEY = os.environ.get("LIBRETRANSLATE_API_KEY", "").strip()


# ----------------------------
# Config
# ----------------------------
@dataclass
class FeedConfig:
    id: str
    name: str
    source_url: str
    item_limit: int = 30
    fetch_full_content: bool = False
    full_content_api_url_template: str = ""
    full_content_api_text_path: str = ""
    full_content_is_html: bool = True
    full_content_api_format: str = "json"
    full_content_extract_sections: List[str] = None


@dataclass
class AppConfig:
    host: str
    port: int
    base_url: str
    translation_provider: str
    source_lang: str
    target_lang: str
    max_chars_per_item: int
    preserve_iocs: bool
    poll_seconds: int
    fetch_full_content: bool
    original_mode: str
    full_content_timeout_seconds: int
    sqlite_path: str
    ttl_seconds: int
    cache_purge_enabled: bool
    image_dir: str
    lt_chunk_chars: int
    lt_timeout_seconds: int
    deepl_chunk_bytes: int
    render_version: str
    feeds: List[FeedConfig]


def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    server = cfg.get("server", {})
    translation = cfg.get("translation", {})
    cache = cfg.get("cache", {})

    feeds_cfg = []
    for fcfg in cfg.get("feeds", []):
        feeds_cfg.append(
            FeedConfig(
                id=str(fcfg["id"]).strip(),
                name=str(fcfg.get("name", fcfg["id"])).strip(),
                source_url=str(fcfg["source_url"]).strip(),
                item_limit=int(fcfg.get("item_limit", 30)),
                fetch_full_content=bool(
                    fcfg.get("fetch_full_content", translation.get("fetch_full_content", False))
                ),
                full_content_api_url_template=str(
                    fcfg.get("full_content_api_url_template", "")
                ).strip(),
                full_content_api_text_path=str(
                    fcfg.get("full_content_api_text_path", "")
                ).strip(),
                full_content_is_html=bool(
                    fcfg.get("full_content_is_html", True)
                ),
                full_content_api_format=str(
                    fcfg.get("full_content_api_format", "json")
                ).strip().lower(),
                full_content_extract_sections=list(
                    fcfg.get("full_content_extract_sections", []) or []
                ),
            )
        )

    return AppConfig(
        host=str(server.get("host", "0.0.0.0")),
        port=int(server.get("port", 8086)),
        base_url=str(server.get("base_url", "")).rstrip("/"),
        translation_provider=str(translation.get("provider", "deepl")).lower(),
        source_lang=str(translation.get("source_lang", "auto")).lower(),
        target_lang=str(translation.get("target_lang", "EN")).upper(),
        max_chars_per_item=int(translation.get("max_chars_per_item", 15000)),
        preserve_iocs=bool(translation.get("preserve_iocs", True)),
        poll_seconds=int(translation.get("poll_seconds", 60 * 60)),
        fetch_full_content=bool(translation.get("fetch_full_content", False)),
        original_mode=str(translation.get("original_mode", "text")).lower(),
        full_content_timeout_seconds=int(
            translation.get("full_content_timeout_seconds", 20)
        ),
        sqlite_path=str(cache.get("sqlite_path", "data/cache.sqlite3")),
        ttl_seconds=int(cache.get("ttl_seconds", 60 * 60 * 24 * 30)),
        cache_purge_enabled=bool(cache.get("purge_enabled", True)),
        image_dir=str(cache.get("image_dir", "data/images")),
        lt_chunk_chars=int(translation.get("lt_chunk_chars", 2000)),
        lt_timeout_seconds=int(translation.get("lt_timeout_seconds", 180)),
        deepl_chunk_bytes=int(translation.get("deepl_chunk_bytes", 120000)),
        render_version=str(cache.get("render_version", "v1")),
        feeds=feeds_cfg,
    )


CFG = load_config()
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "").strip()
if not DEEPL_API_KEY and CFG.translation_provider == "deepl":
    # You can still start the service, but requests will fail until key is set.
    print("WARNING: DEEPL_API_KEY is not set. Translation requests will fail.")
if not LIBRETRANSLATE_API_KEY and CFG.translation_provider == "libretranslate":
    print("WARNING: LIBRETRANSLATE_API_KEY is not set. Translation requests may fail.")


# ----------------------------
# Cache (SQLite)
# ----------------------------
def ensure_db(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS translations (
                cache_key TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                engine TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                src_hash TEXT NOT NULL,
                translated_title TEXT,
                translated_desc TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS feed_cache (
                feed_id TEXT PRIMARY KEY,
                updated_at INTEGER NOT NULL,
                xml TEXT NOT NULL
            )
            """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON translations(created_at)"
        )
        con.commit()


def cache_get(path: str, cache_key: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM translations WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return dict(row) if row else None


def cache_put(
    path: str,
    cache_key: str,
    engine: str,
    target_lang: str,
    src_hash: str,
    translated_title: str,
    translated_desc: str,
) -> None:
    now = int(time.time())
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO translations
            (cache_key, created_at, engine, target_lang, src_hash, translated_title, translated_desc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                now,
                engine,
                target_lang,
                src_hash,
                translated_title,
                translated_desc,
            ),
        )
        con.commit()


def feed_cache_get(path: str, feed_id: str) -> Optional[str]:
    cache_id = f"{feed_id}|{CFG.render_version}"
    with sqlite3.connect(path) as con:
        row = con.execute(
            "SELECT xml FROM feed_cache WHERE feed_id = ?", (cache_id,)
        ).fetchone()
        return row[0] if row else None


def feed_cache_put(path: str, feed_id: str, xml: str) -> None:
    now = int(time.time())
    cache_id = f"{feed_id}|{CFG.render_version}"
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO feed_cache (feed_id, updated_at, xml)
            VALUES (?, ?, ?)
            """,
            (cache_id, now, xml),
        )
        con.commit()


def cache_purge_old(path: str, ttl_seconds: int) -> None:
    cutoff = int(time.time()) - ttl_seconds
    with sqlite3.connect(path) as con:
        con.execute("DELETE FROM translations WHERE created_at < ?", (cutoff,))
        con.commit()


ensure_db(CFG.sqlite_path)


# ----------------------------
# IOC/indicator preservation
# ----------------------------
IOC_PATTERNS = [
    # URLs
    r"https?://[^\s<>\"]+",
    # CVE IDs
    r"\bCVE-\d{4}-\d{4,7}\b",
    # Hex hashes (md5/sha1/sha256 etc. simple heuristic)
    r"\b[a-fA-F0-9]{32}\b",
    r"\b[a-fA-F0-9]{40}\b",
    r"\b[a-fA-F0-9]{64}\b",
    # IPv4
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    # email
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
]


def protect_iocs(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace IOCs with tokens so translation engines are less likely to mangle them.
    Returns protected_text, token_map.
    """
    token_map: Dict[str, str] = {}
    if not text:
        return text, token_map

    combined = re.compile("|".join(f"({p})" for p in IOC_PATTERNS))
    i = 0

    def repl(m: re.Match) -> str:
        nonlocal i
        original = m.group(0)
        token = f"__IOC_{i}__"
        token_map[token] = original
        i += 1
        return token

    return combined.sub(repl, text), token_map


def restore_iocs(text: str, token_map: Dict[str, str]) -> str:
    if not text or not token_map:
        return text
    for token, original in token_map.items():
        text = text.replace(token, original)
    return text


# ----------------------------
# DeepL client
# ----------------------------
async def deepl_translate(texts: List[str], target_lang: str) -> List[str]:
    if not DEEPL_API_KEY:
        raise RuntimeError("DEEPL_API_KEY not set")

    # DeepL supports sending multiple "text" fields
    data = []
    for t in texts:
        data.append(("text", t))
    data.append(("target_lang", target_lang))

    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(DEEPL_ENDPOINT, data=data, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"DeepL error {r.status_code}: {r.text}")

        payload = r.json()
        translations = payload.get("translations", [])
        return [tr.get("text", "") for tr in translations]


# ----------------------------
# RSS -> RSS
# ----------------------------
def text_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def cache_key_for_item(
    feed_id: str, guid_or_link: str, src_hash: str, target_lang: str
) -> str:
    return hashlib.sha256(
        f"{feed_id}|{guid_or_link}|{src_hash}|{target_lang}".encode("utf-8")
    ).hexdigest()


def pick_item_id(entry: Any) -> str:
    # Prefer GUID/id; fallback to link; else title+published
    for k in ("id", "guid", "link"):
        v = getattr(entry, k, None) or entry.get(k)
        if v:
            return str(v)
    title = getattr(entry, "title", "") or entry.get("title", "")
    published = getattr(entry, "published", "") or entry.get("published", "")
    return f"{title}|{published}"


def entry_text(entry: Any) -> Tuple[str, str]:
    title = (getattr(entry, "title", "") or entry.get("title", "") or "").strip()

    # Prefer summary/description
    desc = (getattr(entry, "summary", "") or entry.get("summary", "") or "").strip()
    if not desc:
        desc = (entry.get("description", "") or "").strip()

    # Some feeds use content[]
    if not desc and entry.get("content"):
        try:
            desc = (entry["content"][0].get("value") or "").strip()
        except Exception:
            pass

    # Unescape HTML entities but keep HTML tags intact (RSS readers can render)
    title = html.unescape(title)
    desc = html.unescape(desc)
    return title, desc


def clamp(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def protect_breaks(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    # Preserve paragraph and line breaks through translation.
    text = text.replace("\n\n", "\n<<<PARA>>>\n")
    text = text.replace("\n", "\n<<<LINE>>>\n")
    return text


def restore_breaks(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s*<<<PARA>>>\s*", "\n\n", text)
    text = re.sub(r"\s*<<<LINE>>>\s*", "\n", text)
    return text


def protect_markers(text: str) -> Tuple[str, Dict[str, str]]:
    if not text:
        return "", {}
    tokens: Dict[str, str] = {}

    def make_token() -> str:
        return f"ZZZMARKER{len(tokens)}ZZZ"

    def repl_data_uri(m: re.Match) -> str:
        key = make_token()
        tokens[key] = m.group(0)
        return key

    # Strip large data URIs before translation, restore later.
    text = re.sub(
        r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
        repl_data_uri,
        text,
    )

    def repl(m: re.Match) -> str:
        key = make_token()
        tokens[key] = m.group(0)
        return key

    text = re.sub(
        r"\[\[\[(?:/)?PRE\]\]\]|\[\[\[IMGURL:.*?\]\]\]|\[\[\[IMG:.*?\]\]\]|<<<(?:PARA|LINE)>>>|__IOC_\d+__",
        repl,
        text,
    )
    return text, tokens


def restore_markers(text: str, tokens: Dict[str, str]) -> str:
    if not text or not tokens:
        return text or ""
    # Replace robustly even if translation has added spaces or underscores.
    def replace_marker(match: re.Match) -> str:
        idx = match.group(1)
        key = f"ZZZMARKER{idx}ZZZ"
        return tokens.get(key, match.group(0))

    text = re.sub(
        r"Z\s*Z\s*Z\s*M\s*A\s*R\s*K\s*E\s*R\s*(\d+)\s*Z\s*Z\s*Z",
        replace_marker,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"M\s*A\s*R\s*K\s*E\s*R\s*[_\s]*?(\d+)",
        replace_marker,
        text,
        flags=re.IGNORECASE,
    )
    for key, val in tokens.items():
        text = text.replace(key, val)
    return text


def _render_text_with_pre(text: str, headings: Optional[List[str]] = None) -> str:
    if not text:
        return ""
    heading_set = {h.strip() for h in (headings or []) if h and h.strip()}
    pre_start = "[[[PRE]]]"
    pre_end = "[[[/PRE]]]"

    def render_nonpre(chunk: str, out: List[str]) -> None:
        if not chunk:
            return
        chunk = chunk.replace(pre_start, "").replace(pre_end, "")
        paras = [p for p in re.split(r"\n{2,}", chunk.strip()) if p.strip()]
        for p in paras:
            p = p.strip()
            if p in heading_set:
                out.append("<h3>")
                out.append(html.escape(p))
                out.append("</h3>")
                continue
            if p.endswith(":") and len(p) <= 80 and "\n" not in p:
                out.append("<p><strong>")
                out.append(html.escape(p))
                out.append("</strong></p>")
                continue
            out.append("<p>")
            out.append(html.escape(p).replace("\n", "<br/>"))
            out.append("</p>")

    rendered: List[str] = []
    i = 0
    while i < len(text):
        next_pre = text.find(pre_start, i)
        next_img = re.search(r"\[\[\[IMGURL:.*?\]\]\]|\[\[\[IMG:.*?\]\]\]", text[i:])
        next_img_idx = (i + next_img.start()) if next_img else -1

        # Determine next marker position
        candidates = [pos for pos in (next_pre, next_img_idx) if pos != -1]
        if not candidates:
            render_nonpre(text[i:], rendered)
            break
        next_pos = min(candidates)

        # Render text before the marker
        render_nonpre(text[i:next_pos], rendered)

        if next_pos == next_pre:
            end = text.find(pre_end, next_pre + len(pre_start))
            if end == -1:
                # Unmatched pre; drop marker and continue
                i = next_pre + len(pre_start)
                continue
            pre_body = text[next_pre + len(pre_start) : end]
            rendered.append("<pre>")
            rendered.append(html.escape(pre_body.strip("\n")))
            rendered.append("</pre>")
            i = end + len(pre_end)
        else:
            m = re.match(
                r"\[\[\[IMGURL:(.*?)\]\]\]|\[\[\[IMG:(.*?)\]\]\]",
                text[next_pos:],
            )
            if not m:
                i = next_pos + 1
                continue
            name = m.group(1)
            src = m.group(2)
            if name:
                base = CFG.base_url or ""
                url = f"{base}/images/{name}" if base else f"/images/{name}"
                rendered.append(f"<img src=\"{html.escape(url)}\"/>")
            elif src:
                rendered.append(f"<img src=\"{html.escape(src)}\"/>")
            i = next_pos + m.end(0)

    return "\n".join(rendered).strip()


def build_translated_feed_xml(
    feed_cfg: FeedConfig,
    entries: List[Any],
    translated_title: Dict[int, str],
    translated_desc: Dict[int, str],
    original_title: Dict[int, str],
    original_desc: Dict[int, str],
    original_link: Dict[int, str],
) -> bytes:
    fg = FeedGenerator()
    fg.title(feed_cfg.name)
    fg.link(href=feed_cfg.source_url, rel="alternate")
    fg.description(f"Translated to {CFG.target_lang} from: {feed_cfg.source_url}")
    fg.language("en")

    for idx, entry in enumerate(entries):
        t_title = translated_title.get(idx, "")
        t_desc = translated_desc.get(idx, "")
        if not t_title and not t_desc:
            continue

        item_title = original_title.get(idx, "")
        item_desc = original_desc.get(idx, "")
        item_link = original_link.get(idx, "")

        fe = fg.add_entry()
        fe.id(pick_item_id(entry))

        link = item_link or getattr(entry, "link", None) or entry.get("link") or feed_cfg.source_url
        fe.link(href=str(link))

        pub = getattr(entry, "published_parsed", None) or entry.get("published_parsed")
        if pub:
            pass

        fe.title(t_title or item_title)

        combined = ""
        if t_desc:
            rendered_desc = _render_text_with_pre(
                t_desc, headings=feed_cfg.full_content_extract_sections
            )
            combined += f"<p><strong>English</strong></p>\n{rendered_desc}\n"
        if CFG.original_mode == "text" and item_desc:
            combined += f"\n<hr/>\n<p><strong>Original</strong></p>\n{item_desc}\n"
        elif CFG.original_mode == "link" and link:
            combined += (
                f"\n<hr/>\n<p><strong>Original</strong></p>\n"
                f"<p><a href=\"{link}\">{link}</a></p>\n"
            )
        fe.description(combined or t_desc or "")

    return fg.rss_str(pretty=True)


app = FastAPI(title="RSS Translate Proxy", version="1.0.0")


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}", flush=True)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    dur_ms = int((time.time() - start) * 1000)
    _log(
        f"req method={request.method} path={request.url.path} status={response.status_code} ms={dur_ms}"
    )
    return response


@app.get("/")
def root():
    return {
        "service": "rss-translate-proxy",
        "feeds": [f"/feeds/{f.id}.xml" for f in CFG.feeds],
        "health": "/healthz",
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/images/{name}")
def image(name: str):
    if not re.fullmatch(r"[A-Fa-f0-9]{64}\.[A-Za-z0-9]+", name):
        raise HTTPException(status_code=404, detail="Not found")
    path = os.path.abspath(os.path.join(CFG.image_dir, name))
    if not path.startswith(os.path.abspath(CFG.image_dir) + os.sep):
        raise HTTPException(status_code=404, detail="Not found")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    mime, _ = mimetypes.guess_type(path)
    _log(f"image serve name={name} bytes={os.path.getsize(path)}")
    return FileResponse(path, media_type=mime or "application/octet-stream")


@app.get("/feeds/{feed_id}.xml")
async def translated_feed(feed_id: str):
    feed_cfg = next((f for f in CFG.feeds if f.id == feed_id), None)
    if not feed_cfg:
        raise HTTPException(status_code=404, detail="Unknown feed_id")

    xml = feed_cache_get(CFG.sqlite_path, feed_id)
    if xml is None:
        _log(f"feed_cache miss feed_id={feed_id}")
        xml_bytes = build_translated_feed_xml(feed_cfg, [], {}, {}, {}, {}, {})
    else:
        _log(f"feed_cache hit feed_id={feed_id} bytes={len(xml)}")
        xml_bytes = xml.encode("utf-8")
    return Response(content=xml_bytes, media_type="application/rss+xml; charset=utf-8")
