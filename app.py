import hashlib
import html
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Response
from feedgen.feed import FeedGenerator

DEEPL_ENDPOINT = os.environ.get(
    "DEEPL_ENDPOINT", "https://api-free.deepl.com/v2/translate"
)


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
    target_lang: str
    max_chars_per_item: int
    preserve_iocs: bool
    poll_seconds: int
    fetch_full_content: bool
    original_mode: str
    full_content_timeout_seconds: int
    sqlite_path: str
    ttl_seconds: int
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
        feeds=feeds_cfg,
    )


CFG = load_config()
DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "").strip()
if not DEEPL_API_KEY:
    # You can still start the service, but requests will fail until key is set.
    print("WARNING: DEEPL_API_KEY is not set. Translation requests will fail.")


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
    with sqlite3.connect(path) as con:
        row = con.execute(
            "SELECT xml FROM feed_cache WHERE feed_id = ?", (feed_id,)
        ).fetchone()
        return row[0] if row else None


def feed_cache_put(path: str, feed_id: str, xml: str) -> None:
    now = int(time.time())
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO feed_cache (feed_id, updated_at, xml)
            VALUES (?, ?, ?)
            """,
            (feed_id, now, xml),
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
            combined += f"<p><strong>English</strong></p>\n{t_desc}\n"
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


@app.get("/feeds/{feed_id}.xml")
async def translated_feed(feed_id: str):
    feed_cfg = next((f for f in CFG.feeds if f.id == feed_id), None)
    if not feed_cfg:
        raise HTTPException(status_code=404, detail="Unknown feed_id")

    xml = feed_cache_get(CFG.sqlite_path, feed_id)
    if xml is None:
        xml_bytes = build_translated_feed_xml(feed_cfg, [], {}, {}, {}, {}, {})
    else:
        xml_bytes = xml.encode("utf-8")
    return Response(content=xml_bytes, media_type="application/rss+xml; charset=utf-8")
