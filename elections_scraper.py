#!/usr/bin/env python3
"""
Ayn - US Elections 2026 scraper.

Every run:
  1. Search the approved sources (sources.json) using the keywords (keywords.json)
     via Google News RSS, GDELT and the sources' own RSS feeds.
  2. Check 1 (keywords): items must match the election keywords.
  3. Check 2 (content): Claude reads title + snippet, keeps only items that are
     genuinely about US elections (news, plus substantive analysis only),
     and links each item to an existing story from the last 7 days if it is the same story.
  4. Save to Supabase. Each story's source_count grows as new sources report it.

Environment variables (GitHub Secrets):
  SUPABASE_URL, SUPABASE_SERVICE_KEY, ANTHROPIC_API_KEY
Optional:
  CLAUDE_MODEL (default: claude-haiku-4-5-20251001)
"""

import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlparse
from zoneinfo import ZoneInfo

import anthropic
import feedparser
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ["SUPABASE_SERVICE_KEY"]
MODEL = os.environ.get("CLAUDE_MODEL") or "claude-haiku-4-5-20251001"

UTC = timezone.utc
RIYADH = ZoneInfo("Asia/Riyadh")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
BATCH_SIZE = 25          # candidates per Claude call
MATCH_DAYS = 7           # compare new items with stories from the last 7 days
MAX_EXISTING = 400       # max existing stories sent to Claude per call
MAX_CANDIDATES = 1500    # safety cap per run
GOOGLE_PAUSE = 1.0       # seconds between Google News requests
GDELT_PAUSE = 5.5        # GDELT allows about one request every 5 seconds


def log(*args):
    print(*args, flush=True)


def load_json(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ Supabase

def sb(method, path, params=None, body=None, prefer=None):
    headers = {"apikey": SB_KEY, "Content-Type": "application/json"}
    if SB_KEY.startswith("eyJ"):  # legacy JWT keys also go in Authorization
        headers["Authorization"] = f"Bearer {SB_KEY}"
    if prefer:
        headers["Prefer"] = prefer
    r = requests.request(
        method, f"{SB_URL}/rest/v1/{path}", headers=headers, params=params,
        data=json.dumps(body) if body is not None else None, timeout=60,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase {method} {path} -> {r.status_code}: {r.text[:300]}")
    return r.json() if r.text.strip() else None


def sb_all(path, params):
    rows, offset = [], 0
    while True:
        chunk = sb("GET", path, dict(params, limit=1000, offset=offset)) or []
        rows.extend(chunk)
        if len(chunk) < 1000:
            return rows
        offset += 1000


# ------------------------------------------------------------------ helpers

def iso(dt):
    return dt.astimezone(UTC).isoformat()


def riyadh_date(dt):
    return dt.astimezone(RIYADH).date().isoformat()


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def strip_tags(s):
    s = re.sub(r"<[^>]+>", " ", html.unescape(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def norm_title(t):
    t = html.unescape(t or "").lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()[:140]


def host_of(url):
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def to_dt(struct):
    if not struct:
        return None
    try:
        return datetime(*struct[:6], tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def build_source_index(sources):
    idx = [(d.lower(), s["name"]) for s in sources for d in s["domains"]]
    idx.sort(key=lambda x: -len(x[0]))
    return idx


def source_for(host, idx):
    for domain, name in idx:
        if host == domain or host.endswith("." + domain):
            return name
    return None


def term_chunks(terms, max_len=340):
    """Split terms into OR-groups short enough for one search query."""
    quoted = [f'"{t}"' if " " in t else t for t in terms]
    chunks, cur = [], []
    for q in quoted:
        if cur and len(" OR ".join(cur + [q])) > max_len:
            chunks.append(cur)
            cur = [q]
        else:
            cur.append(q)
    if cur:
        chunks.append(cur)
    return ["(" + " OR ".join(c) + ")" for c in chunks]


def keyword_regex(terms):
    parts = [r"\b" + re.escape(t.lower()) + r"\b" for t in terms if t.strip()]
    return re.compile("|".join(parts), re.I)


# ------------------------------------------------------------------ fetchers

def fetch_google(domain, query_chunk, since):
    after = (since - timedelta(days=1)).strftime("%Y-%m-%d")
    q = f"{query_chunk} site:{domain} after:{after}"
    url = ("https://news.google.com/rss/search?q=" + quote(q)
           + "&hl=en-US&gl=US&ceid=US:en")
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for e in feedparser.parse(r.content).entries:
        src = e.get("source") or {}
        src_title = src.get("title", "") if isinstance(src, dict) else ""
        src_href = src.get("href", "") if isinstance(src, dict) else ""
        title = e.get("title", "")
        suffix = " - " + src_title
        if src_title and title.endswith(suffix):
            title = title[: -len(suffix)]
        out.append({
            "title": title.strip(),
            "link": e.get("link", ""),
            "host": host_of(src_href) or domain,
            "published": to_dt(e.get("published_parsed")),
            "snippet": "",
            "origin": "google",
        })
    return out


def fetch_gdelt(domain, query_chunk, since):
    params = {
        "query": f"{query_chunk} domain:{domain} sourcelang:english",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": "250",
        "sort": "DateDesc",
        "startdatetime": since.astimezone(UTC).strftime("%Y%m%d%H%M%S"),
    }
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc",
                     params=params, headers=HEADERS, timeout=45)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    try:
        data = r.json()
    except ValueError:
        raise RuntimeError(r.text[:120].strip())
    out = []
    for a in data.get("articles", []) or []:
        try:
            pub = datetime.strptime(a.get("seendate", ""), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            pub = None
        out.append({
            "title": a.get("title", ""),
            "link": a.get("url", ""),
            "host": host_of(a.get("url", "")),
            "published": pub,
            "snippet": "",
            "origin": "gdelt",
        })
    return out


def fetch_rss(url, source_name):
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for e in feedparser.parse(r.content).entries:
        out.append({
            "title": strip_tags(e.get("title", "")),
            "link": e.get("link", ""),
            "source": source_name,
            "published": to_dt(e.get("published_parsed") or e.get("updated_parsed")),
            "snippet": strip_tags(e.get("summary", ""))[:500],
            "origin": "rss",
        })
    return out


META_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?:og:description|description)["\'][^>]*>', re.I)
CONTENT_RE = re.compile(r'content=["\']([^"\']*)["\']', re.I)


def fetch_description(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return ""
        m = META_RE.search(r.text[:300000])
        if not m:
            return ""
        c = CONTENT_RE.search(m.group(0))
        return strip_tags(c.group(1))[:500] if c else ""
    except requests.RequestException:
        return ""


# ------------------------------------------------------------------ Claude

SYSTEM_PROMPT = """You screen items for a monitoring page about the 2026 United States elections, used by policy analysts.

For every CANDIDATE decide:
1. keep: true only if the item is primarily about US elections: the 2026 midterms, Senate/House/governor/state races, primaries, candidates and campaigns, polling, voters and turnout, voting rules and election administration, redistricting, election results, or early positioning for the 2028 presidential race. Set keep=false when elections are only mentioned in passing, when it is about a non-US election, or when it is a live-blog fragment, video or photo listing, newsletter digest, quiz, or other low-value item. Opinion and analysis pieces are kept only when they offer substantive analysis (a clear argument backed by data, evidence or expert insight), not thin commentary.
2. match: if the item covers the same specific story as one of the EXISTING STORIES, return that story's id. Same story means the same specific event, development, announcement, poll release or claim, not merely the same broad topic or the same race.
3. group: if match is null, give candidates in this batch that cover the same specific story the same short label (for example "g1"); otherwise null.

Judge from the title, source and snippet. Return JSON only, no prose, in exactly this shape:
{"results":[{"i":0,"keep":true,"match":null,"group":"g1"}]}
Include one entry for every candidate index."""


def judge(client, batch, stories):
    existing = "\n".join(f"{s['id']} | {s['title']}" for s in stories[:MAX_EXISTING]) or "(none)"
    lines = []
    for i, it in enumerate(batch):
        line = f"[{i}] {it['source']} | {it['title']}"
        if it.get("snippet"):
            line += f" | {it['snippet'][:300]}"
        lines.append(line)
    content = f"EXISTING STORIES (id | title):\n{existing}\n\nCANDIDATES:\n" + "\n".join(lines)

    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=4096, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start:end + 1])
            return {int(r["i"]): r for r in data.get("results", []) if "i" in r}
        except Exception as ex:  # noqa: BLE001 - retry any API/parse failure
            log(f"  Claude attempt {attempt + 1} failed: {ex}")
            time.sleep(5 * (attempt + 1))
    return None


# ------------------------------------------------------------------ storage

def insert_articles(story_id, arts):
    body = [{
        "story_id": story_id,
        "title": a["title"],
        "source": a["source"],
        "link": a["link"],
        "published_at": iso(a["published"]),
    } for a in arts]
    sb("POST", "election_articles", params={"on_conflict": "link"}, body=body,
       prefer="resolution=ignore-duplicates,return=minimal")


def refresh_story(story_id):
    rows = sb("GET", "election_articles", {
        "select": "title,source,link,published_at",
        "story_id": f"eq.{story_id}",
        "order": "published_at.asc",
    }) or []
    if not rows:
        return
    first = rows[0]
    sb("PATCH", "election_stories", params={"id": f"eq.{story_id}"}, body={
        "title": first["title"],
        "source": first["source"],
        "link": first["link"],
        "published_at": first["published_at"],
        "story_date": riyadh_date(parse_ts(first["published_at"])),
        "source_count": len({r["source"] for r in rows}),
        "updated_at": iso(datetime.now(UTC)),
    }, prefer="return=minimal")


def create_story(arts):
    arts = sorted(arts, key=lambda a: a["published"])
    first = arts[0]
    row = sb("POST", "election_stories", body={
        "title": first["title"],
        "source": first["source"],
        "link": first["link"],
        "published_at": iso(first["published"]),
        "story_date": riyadh_date(first["published"]),
        "source_count": len({a["source"] for a in arts}),
        "updated_at": iso(datetime.now(UTC)),
    }, prefer="return=representation")[0]
    insert_articles(row["id"], arts)
    return row


def mark_seen(keys):
    if keys:
        sb("POST", "election_seen", params={"on_conflict": "key"},
           body=[{"key": k} for k in keys],
           prefer="resolution=ignore-duplicates,return=minimal")


# ------------------------------------------------------------------ main

def main():
    kw = load_json("keywords.json")
    sources = load_json("sources.json")
    src_idx = build_source_index(sources)
    terms = [t for t in kw.get("search_terms", []) + kw.get("names", []) if t.strip()]
    kw_re = keyword_regex(terms)
    chunks = term_chunks(terms)

    now = datetime.now(UTC)
    backfill = datetime.fromisoformat(kw["backfill_start"]).replace(tzinfo=RIYADH).astimezone(UTC)
    since = max(backfill, now - timedelta(days=int(kw.get("lookback_days", 5))))
    log(f"Model: {MODEL} | window starts {iso(since)} | {len(terms)} terms, {len(chunks)} query groups")

    seen = {r["key"] for r in sb_all("election_seen", {
        "select": "key", "created_at": f"gte.{iso(now - timedelta(days=30))}"})}
    known_links = {r["link"] for r in sb_all("election_articles", {
        "select": "link", "created_at": f"gte.{iso(now - timedelta(days=30))}"})}
    log(f"Already processed: {len(seen)} items, {len(known_links)} saved links")

    raw = []

    # Direct RSS feeds (Check 1 applied below because feeds are not keyword-searched)
    for s in sources:
        for url in s.get("rss", []):
            try:
                items = fetch_rss(url, s["name"])
                items = [it for it in items if kw_re.search(f"{it['title']} {it['snippet']}")]
                raw.extend(items)
                log(f"RSS  {s['name']}: {len(items)} matching")
            except Exception as ex:  # noqa: BLE001
                log(f"RSS  {s['name']} failed: {ex}")

    # Google News + GDELT (keyword search itself is Check 1)
    for s in sources:
        for domain in s["domains"]:
            for chunk in chunks:
                try:
                    items = fetch_google(domain, chunk, since)
                    raw.extend(items)
                    log(f"GNEWS {domain}: {len(items)}")
                except Exception as ex:  # noqa: BLE001
                    log(f"GNEWS {domain} failed: {ex}")
                time.sleep(GOOGLE_PAUSE)
            try:
                items = fetch_gdelt(domain, chunks[0], since)
                raw.extend(items)
                log(f"GDELT {domain}: {len(items)}")
            except Exception as ex:  # noqa: BLE001
                log(f"GDELT {domain} failed: {ex}")
            time.sleep(GDELT_PAUSE)

    # Normalize, whitelist, window, de-duplicate
    candidates = {}
    for it in raw:
        if not it.get("title") or not it.get("link"):
            continue
        name = it.get("source") or source_for(it.get("host", ""), src_idx)
        if not name:
            continue
        it["source"] = name
        it["published"] = it.get("published") or now
        if it["published"] < since or it["published"] > now + timedelta(hours=1):
            continue
        key = f"{name}|{norm_title(it['title'])}"
        if key in seen or it["link"] in known_links:
            continue
        it["key"] = key
        prev = candidates.get(key)
        if prev is None or (prev["origin"] == "google" and it["origin"] != "google"):
            if prev and not it.get("snippet"):
                it["snippet"] = prev.get("snippet", "")
            candidates[key] = it

    items = sorted(candidates.values(), key=lambda x: x["published"])[:MAX_CANDIDATES]
    log(f"New candidates after Check 1: {len(items)}")
    if not items:
        log("Nothing new.")
        return

    # Add page descriptions for direct links that have no snippet
    need = [it for it in items if not it.get("snippet") and it["origin"] != "google"]
    if need:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for it, desc in zip(need, pool.map(lambda x: fetch_description(x["link"]), need)):
                it["snippet"] = desc
        log(f"Fetched descriptions for {len(need)} items")

    stories = sb_all("election_stories", {
        "select": "id,title,updated_at",
        "updated_at": f"gte.{iso(now - timedelta(days=MATCH_DAYS))}",
        "order": "updated_at.desc",
    })
    story_ids = {s["id"] for s in stories}
    log(f"Existing stories in the last {MATCH_DAYS} days: {len(stories)}")

    client = anthropic.Anthropic()
    kept = created = attached = 0

    for b in range(0, len(items), BATCH_SIZE):
        batch = items[b:b + BATCH_SIZE]
        results = judge(client, batch, stories)
        if results is None:
            log(f"Batch {b // BATCH_SIZE + 1}: skipped, will retry next run")
            continue

        groups, singles, matches = {}, [], {}
        for i, it in enumerate(batch):
            r = results.get(i)
            if not r or not r.get("keep"):
                continue
            kept += 1
            match = r.get("match")
            try:
                match = int(match) if match is not None else None
            except (TypeError, ValueError):
                match = None
            if match in story_ids:
                matches.setdefault(match, []).append(it)
            elif r.get("group"):
                groups.setdefault(str(r["group"]), []).append(it)
            else:
                singles.append([it])

        for sid, arts in matches.items():
            insert_articles(sid, arts)
            refresh_story(sid)
            attached += len(arts)

        for arts in list(groups.values()) + singles:
            row = create_story(arts)
            created += 1
            story_ids.add(row["id"])
            stories.insert(0, {"id": row["id"], "title": row["title"]})

        mark_seen([it["key"] for it in batch])
        log(f"Batch {b // BATCH_SIZE + 1}: {len(batch)} checked")

    log(f"Done. Kept {kept} | new stories {created} | added to existing stories {attached}")


if __name__ == "__main__":
    main()
