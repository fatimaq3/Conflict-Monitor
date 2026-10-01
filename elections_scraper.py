#!/usr/bin/env python3
"""
Ayn - US Elections 2026: selected analytical articles.

Every run (every 4 hours):
  1. Search the approved sources (sources.json) with the keywords (keywords.json)
     via Google News RSS and the sources' own RSS feeds.        (Check 1: keywords)
  2. Triage (Claude Haiku): keep only items that may be analytical pieces about US elections.
  3. Final selection (Claude Sonnet): score analytical value, keep only strong pieces
     (about 5-10 a day), and link each to an existing issue if it analyzes the same issue.
  4. Save to Supabase. Each issue's count grows as more articles analyze it.

Environment variables (GitHub Secrets):
  SUPABASE_URL, SUPABASE_SERVICE_KEY, ANTHROPIC_API_KEY
Optional:
  CLAUDE_MODEL (default claude-sonnet-5), CLAUDE_TRIAGE_MODEL (default claude-haiku-4-5-20251001)
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
MODEL = os.environ.get("CLAUDE_MODEL") or "claude-sonnet-5"
TRIAGE_MODEL = os.environ.get("CLAUDE_TRIAGE_MODEL") or "claude-haiku-4-5-20251001"

UTC = timezone.utc
RIYADH = ZoneInfo("Asia/Riyadh")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
BATCH_SIZE = 20          # candidates per final-selection call
TRIAGE_BATCH = 50        # titles per triage call
TRIAGE_WORKERS = 6       # parallel triage calls
TIME_BUDGET_MIN = 45     # stop AI work after this many minutes; the rest waits for the next run
MATCH_DAYS = 7           # compare new items with issues from the last 7 days
MAX_EXISTING = 400       # max existing issues sent to Claude per call
MAX_CANDIDATES = 8000    # safety cap per run
GOOGLE_PAUSE = 1.0       # seconds between Google News requests
GOOGLE_QUERY_LEN = 80    # short OR-groups: long queries make Google ignore site:
SKIP_TITLE_RE = re.compile(
    r"^\s*(watch|video|live|listen|photos?|exclusive|breaking)\s*[:|]|live updates|week in politics|"
    r"in brief|toplines|cross-tabs|primary results|election results|voter guide|what to know|countdown|"
    r"newsletter|podcast|\bpoll finds\b", re.I)


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
    if r.status_code in (401, 403) or '"42501"' in r.text:
        raise RuntimeError(
            f"Supabase {method} {path} -> {r.status_code}: permission denied. "
            "SUPABASE_SERVICE_KEY must be the service_role / secret key, not the anon / publishable key. "
            f"Details: {r.text[:200]}")
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

def fetch_google(domain, query_chunk, days):
    q = f"site:{domain} {query_chunk} when:{days}d"
    url = ("https://news.google.com/rss/search?q=" + quote(q)
           + "&hl=en-US&gl=US&ceid=US:en")
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for e in feedparser.parse(r.content).entries:
        src = e.get("source") or {}
        src_title = src.get("title", "") if isinstance(src, dict) else ""
        src_href = (src.get("href") or src.get("url") or "") if isinstance(src, dict) else ""
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



# ------------------------------------------------------------------ Claude

TRIAGE_PROMPT = """You triage items for a page that lists only ANALYTICAL articles about the 2026 United States elections.

For each candidate, decide whether it could be an analytical piece (analysis, opinion or op-ed, explainer, or long-form feature built around an argument) that is primarily about US elections: the 2026 midterms, specific races, campaigns and candidates, voters and polling trends, voting rules, election administration, redistricting, or the 2028 race.

Answer NO for: straight news reports and breaking news, poll toplines or cross-tabs, forecast or data pages, results pages, voter guides, race trackers, live blogs, videos, podcasts, newsletters, roundups, digests and "in brief" items, non-English items, and anything not primarily about US elections.
When unsure whether a relevant piece is analysis, answer YES; a stricter editor reads the full content next.

Return JSON only: {"keep":[0,3,7]}"""


SYSTEM_PROMPT = """You are the final editor of a page listing only the best ANALYTICAL articles about the 2026 United States elections, read by senior policy analysts. The page shows roughly 5 to 10 articles per day across all sources, so be highly selective.

For every CANDIDATE return:
1. analysis: true only for a genuine analytical piece (analysis, opinion or op-ed, explainer, or long-form feature built around an argument). False for news reports, roundups, digests, "in brief" items, race lists or "races to watch", trackers, poll toplines, forecast pages, guides, videos, podcasts and non-English items.
2. score (1-10): analytical value to a policy analyst. Weigh depth of argument, use of evidence and data, originality of insight, significance for election outcomes or the direction of US politics, and credibility. 9-10 exceptional and essential; 8 strong and clearly worth reading; 6-7 decent but not essential; 5 or below thin. Partisan attack pieces, pieces built on one quote, and celebrity or human-interest angles score 5 or below.
3. match: if the piece clearly analyzes the same specific issue or question as one of the EXISTING ISSUES, return that id. Different angles on the same specific question count as the same issue (for example, several analyses of why Republican candidates are distancing themselves from Trump). Sharing a word or a broad theme ("races", "polls", "the midterms", "Trump's popularity") is NOT enough. If the title does not make the specific question clear, return null. When in doubt, return null.
4. group: if match is null, give candidates in this batch that clearly analyze the same specific issue the same short label (for example "g1"); otherwise null. Same strict rule.

Judge from the title, source and snippet. Return JSON only, no prose, in exactly this shape:
{"results":[{"i":0,"analysis":true,"score":8,"match":null,"group":null}]}
Include one entry for every candidate index."""


def ask_claude(client, model, system, content, max_tokens=4096):
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model, max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            start, end = text.find("{"), text.rfind("}")
            return json.loads(text[start:end + 1])
        except Exception as ex:  # noqa: BLE001 - retry any API/parse failure
            log(f"  Claude ({model}) attempt {attempt + 1} failed: {ex}")
            time.sleep(5 * (attempt + 1))
    return None


def candidate_lines(batch):
    lines = []
    for i, it in enumerate(batch):
        line = f"[{i}] {it['source']} | {it['title']}"
        if it.get("snippet"):
            line += f" | {it['snippet'][:300]}"
        lines.append(line)
    return "\n".join(lines)


def triage(client, batch):
    data = ask_claude(client, TRIAGE_MODEL, TRIAGE_PROMPT,
                      "CANDIDATES:\n" + candidate_lines(batch), max_tokens=1024)
    if data is None:
        return None
    keep = set()
    for i in data.get("keep", []):
        try:
            keep.add(int(i))
        except (TypeError, ValueError):
            pass
    return keep


def judge(client, batch, stories):
    existing = "\n".join(f"{s['id']} | {s['title']}" for s in stories[:MAX_EXISTING]) or "(none)"
    content = [
        {"type": "text", "text": f"EXISTING ISSUES (id | title of first article):\n{existing}",
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"CANDIDATES:\n{candidate_lines(batch)}"},
    ]
    data = ask_claude(client, MODEL, SYSTEM_PROMPT, content)
    if data is None:
        return None
    return {int(r["i"]): r for r in data.get("results", []) if "i" in r}


# ------------------------------------------------------------------ storage

def insert_articles(story_id, arts):
    body = [{
        "story_id": story_id,
        "title": a["title"],
        "source": a["source"],
        "link": a["link"],
        "published_at": iso(a["published"]),
        "score": a.get("score"),
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
        "source_count": len(rows),
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
        "source_count": len(arts),
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

START = datetime.now(UTC)
MIN_SCORE = 8
DAILY_CAP = 10


def main():
    global MIN_SCORE, DAILY_CAP
    kw = load_json("keywords.json")
    MIN_SCORE = float(kw.get("min_score", MIN_SCORE))
    DAILY_CAP = int(kw.get("daily_cap", DAILY_CAP))
    sources = load_json("sources.json")
    src_idx = build_source_index(sources)
    terms = [t for t in kw.get("search_terms", []) + kw.get("names", []) if t.strip()]
    kw_re = keyword_regex(terms)
    google_chunks = term_chunks(terms, max_len=GOOGLE_QUERY_LEN)

    now = datetime.now(UTC)
    backfill = datetime.fromisoformat(kw["backfill_start"]).replace(tzinfo=RIYADH).astimezone(UTC)
    since = max(backfill, now - timedelta(days=int(kw.get("lookback_days", 5))))
    days = max(1, int((now - since).total_seconds() // 86400) + 1)
    log(f"Models: triage {TRIAGE_MODEL}, final {MODEL} | min score {MIN_SCORE}, "
        f"daily cap {DAILY_CAP} | window starts {iso(since)}")

    seen = {r["key"] for r in sb_all("election_seen", {
        "select": "key", "created_at": f"gte.{iso(now - timedelta(days=30))}"})}
    known_links = {r["link"] for r in sb_all("election_articles", {
        "select": "link", "created_at": f"gte.{iso(now - timedelta(days=30))}"})}
    log(f"Already processed: {len(seen)} items, {len(known_links)} saved links")

    raw = []
    for s in sources:
        for url in s.get("rss", []):
            try:
                items = fetch_rss(url, s["name"])
                items = [it for it in items if kw_re.search(f"{it['title']} {it['snippet']}")]
                raw.extend(items)
                log(f"RSS  {s['name']}: {len(items)} matching")
            except Exception as ex:  # noqa: BLE001
                log(f"RSS  {s['name']} failed: {ex}")

    for s in sources:
        for domain in s["domains"]:
            got = on_site = 0
            for chunk in google_chunks:
                try:
                    items = fetch_google(domain, chunk, days)
                    raw.extend(items)
                    got += len(items)
                    on_site += sum(1 for it in items if source_for(it["host"], src_idx) == s["name"])
                except Exception as ex:  # noqa: BLE001
                    log(f"GNEWS {domain} failed: {ex}")
                time.sleep(GOOGLE_PAUSE)
            log(f"GNEWS {domain}: {got} results, {on_site} from this source")

    candidates = {}
    drops = {"empty_or_video": 0, "not_approved_source": 0, "outside_window": 0, "already_processed": 0}
    for it in raw:
        if not it.get("title") or not it.get("link") or SKIP_TITLE_RE.search(it["title"]):
            drops["empty_or_video"] += 1
            continue
        name = it.get("source") or source_for(it.get("host", ""), src_idx)
        if not name:
            drops["not_approved_source"] += 1
            continue
        it["source"] = name
        it["published"] = it.get("published") or now
        if it["published"] < since or it["published"] > now + timedelta(hours=1):
            drops["outside_window"] += 1
            continue
        key = f"{name}|{norm_title(it['title'])}"
        if key in seen or it["link"] in known_links:
            drops["already_processed"] += 1
            continue
        it["key"] = key
        prev = candidates.get(key)
        if prev is None or (prev["origin"] == "google" and it["origin"] != "google"):
            if prev and not it.get("snippet"):
                it["snippet"] = prev.get("snippet", "")
            candidates[key] = it

    log(f"Fetched {len(raw)} raw items | dropped: {drops}")
    items = sorted(candidates.values(), key=lambda x: x["published"])[:MAX_CANDIDATES]
    log(f"New candidates after Check 1: {len(items)}")
    if not items:
        log("Nothing new.")
        return

    client = anthropic.Anthropic()
    deadline = START + timedelta(minutes=TIME_BUDGET_MIN)

    batches = [items[b:b + TRIAGE_BATCH] for b in range(0, len(items), TRIAGE_BATCH)]
    shortlisted = []
    with ThreadPoolExecutor(max_workers=TRIAGE_WORKERS) as pool:
        for batch, keep in zip(batches, pool.map(lambda bt: triage(client, bt), batches)):
            if keep is None:
                continue
            shortlisted.extend(it for i, it in enumerate(batch) if i in keep)
            mark_seen([it["key"] for i, it in enumerate(batch) if i not in keep])
    log(f"Shortlisted as possible analysis: {len(shortlisted)}")

    stories = sb_all("election_stories", {
        "select": "id,title,updated_at",
        "updated_at": f"gte.{iso(now - timedelta(days=MATCH_DAYS))}",
        "order": "updated_at.desc",
    })
    story_ids = {s["id"] for s in stories}
    day_counts = {}
    for r in sb_all("election_stories", {"select": "story_date",
                                         "story_date": f"gte.{riyadh_date(since)}"}):
        day_counts[r["story_date"]] = day_counts.get(r["story_date"], 0) + 1
    log(f"Existing issues in the last {MATCH_DAYS} days: {len(stories)}")

    kept = created = attached = 0
    for b in range(0, len(shortlisted), BATCH_SIZE):
        if datetime.now(UTC) > deadline:
            log("Time budget reached; remaining shortlisted items wait for the next run")
            break
        batch = shortlisted[b:b + BATCH_SIZE]
        results = judge(client, batch, stories)
        if results is None:
            log(f"Batch {b // BATCH_SIZE + 1}: skipped, will retry next run")
            continue

        groups, singles, matches = {}, [], {}
        for i, it in enumerate(batch):
            r = results.get(i)
            if not r or not r.get("analysis"):
                continue
            try:
                score = float(r.get("score") or 0)
            except (TypeError, ValueError):
                score = 0
            if score < MIN_SCORE:
                continue
            it["score"] = score
            try:
                match = int(r["match"]) if r.get("match") is not None else None
            except (TypeError, ValueError):
                match = None
            if match in story_ids:
                matches.setdefault(match, []).append(it)
                kept += 1
                continue
            day = riyadh_date(it["published"])
            if day_counts.get(day, 0) >= DAILY_CAP and score < 9:
                continue
            kept += 1
            if r.get("group"):
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
            day_counts[row["story_date"]] = day_counts.get(row["story_date"], 0) + 1

        mark_seen([it["key"] for it in batch])
        log(f"Batch {b // BATCH_SIZE + 1}: {len(batch)} reviewed")

    log(f"Done. Kept {kept} | new issues {created} | added to existing issues {attached}")


if __name__ == "__main__":
    main()
