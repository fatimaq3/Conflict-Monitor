#!/usr/bin/env python3
"""
Ayn - US Elections 2026: selected analytical articles.

Every run (every 4 hours):
  1. Search the approved sources (sources.json) with the keywords (keywords.json)
     via Google News RSS and the sources' own RSS feeds.        (Check 1: keywords)
  2. Triage by title (Claude Haiku): drop straight news and non-analysis.
  3. Resolve the original article link and read its content (description + opening
     text). No readable content means no publishing.
  4. Editor pass (Claude Sonnet): article type, the analytical question it addresses
     (in Arabic), score out of 10, and whether it belongs to an existing issue.
  5. Independent verification pass (Claude Sonnet): an article is published, and an
     article is filed under an issue, only when both passes agree.
  6. Save to Supabase with the original link.

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
import trafilatura

try:
    from googlenewsdecoder import gnewsdecoder as _gdecode
except ImportError:
    try:
        from googlenewsdecoder import new_decoderv1 as _gdecode
    except ImportError:
        _gdecode = None

ROOT = os.path.dirname(os.path.abspath(__file__))
SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ["SUPABASE_SERVICE_KEY"]
MODEL = os.environ.get("CLAUDE_MODEL") or "claude-sonnet-5"
TRIAGE_MODEL = os.environ.get("CLAUDE_TRIAGE_MODEL") or "claude-haiku-4-5-20251001"

UTC = timezone.utc
RIYADH = ZoneInfo("Asia/Riyadh")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
TRIAGE_BATCH = 50        # titles per triage call
TRIAGE_WORKERS = 6       # parallel triage calls
BATCH_SIZE = 10          # articles (with content) per editor / verifier call
CONTENT_WORDS = 800      # words of article text given to the editor
MIN_CONTENT_WORDS = 35   # below this, the article is not published
TIME_BUDGET_MIN = 45     # stop AI work after this many minutes; the rest waits for the next run
MATCH_DAYS = 7           # compare new articles with issues from the last 7 days
MAX_EXISTING = 300       # max existing issues sent per call
MAX_CANDIDATES = 8000    # safety cap per run
GOOGLE_PAUSE = 1.0       # seconds between Google News requests
GOOGLE_QUERY_LEN = 80    # short OR-groups: long queries make Google ignore site:
ANALYTIC_TYPES = {"analysis", "opinion", "explainer", "feature"}
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


# ------------------------------------------------------------------ content

META_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?:og:description|description|twitter:description)["\'][^>]*>', re.I)
CONTENT_RE = re.compile(r'content=["\']([^"\']*)["\']', re.I)


def resolve_link(url):
    """Return (publisher URL or None, error message) for a Google News link."""
    if "news.google.com" not in url:
        return url, ""
    if _gdecode is None:
        return None, "decoder not installed"
    err = ""
    for attempt in range(2):
        try:
            try:
                res = _gdecode(url, interval=1)
            except TypeError:
                res = _gdecode(url)
            if isinstance(res, dict) and res.get("status") and res.get("decoded_url"):
                return res["decoded_url"], ""
            err = str(res.get("message") if isinstance(res, dict) else res)[:150]
        except Exception as ex:  # noqa: BLE001
            err = str(ex)[:150]
        time.sleep(2)
    return None, err


def fetch_content(url):
    """(description + opening text, status label)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as ex:
        return "", type(ex).__name__
    if r.status_code != 200 or not r.text:
        return "", f"HTTP {r.status_code}"
    page = r.text[:2_000_000]
    desc = ""
    m = META_RE.search(page[:300000])
    if m:
        c = CONTENT_RE.search(m.group(0))
        desc = strip_tags(c.group(1)) if c else ""
    try:
        body = trafilatura.extract(page, include_comments=False, include_tables=False,
                                   favor_precision=True) or ""
    except Exception:  # noqa: BLE001
        body = ""
    text = " ".join(body.split()[:CONTENT_WORDS])
    if desc and desc[:60] not in text:
        text = (desc + "\n\n" + text).strip()
    return text, ("ok" if len(text.split()) >= MIN_CONTENT_WORDS else "too short")


def load_content(items):
    """Resolve links (sequential: Google rate-limits) then fetch pages (parallel). Logs a diagnosis."""
    decode_fail, decode_errors = 0, {}
    for it in items:
        if it["origin"] == "google":
            it["url"], err = resolve_link(it["link"])
            if not it["url"]:
                decode_fail += 1
                decode_errors[err] = decode_errors.get(err, 0) + 1
            time.sleep(0.5)
        else:
            it["url"] = it["link"]
    todo = [it for it in items if it.get("url")]
    status_by_source = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for it, (text, status) in zip(todo, pool.map(lambda x: fetch_content(x["url"]), todo)):
            it["content"] = text
            per = status_by_source.setdefault(it["source"], {})
            per[status] = per.get(status, 0) + 1
    for it in items:
        it.setdefault("content", "")
        if it["origin"] == "rss" and it.get("snippet") and it["snippet"][:60] not in it["content"]:
            it["content"] = (it["snippet"] + "\n\n" + it["content"]).strip()
    log(f"Link resolving: {len(items) - decode_fail} ok, {decode_fail} failed")
    for err, n in sorted(decode_errors.items(), key=lambda x: -x[1])[:3]:
        log(f"  decode error x{n}: {err}")
    for src, per in sorted(status_by_source.items()):
        log(f"  page fetch {src}: {per}")
    return [it for it in items if len(it["content"].split()) >= MIN_CONTENT_WORDS]


# ------------------------------------------------------------------ Claude

TRIAGE_PROMPT = """You triage items for a page that lists only ANALYTICAL articles about the 2026 United States elections.

For each candidate, decide whether it could be an analytical piece (analysis, opinion or op-ed, explainer, or long-form feature built around an argument) that is primarily about US elections: the 2026 midterms, specific races, campaigns and candidates, voters and polling trends, voting rules, election administration, redistricting, or the 2028 race.

Answer NO for: straight news reports and breaking news, poll toplines or cross-tabs, forecast or data pages, results pages, voter guides, race trackers, live blogs, videos, podcasts, newsletters, roundups, digests and "in brief" items, non-English items, and anything not primarily about US elections.
When unsure whether a relevant piece is analysis, answer YES; a stricter editor reads the full content next.

Return JSON only: {"keep":[0,3,7]}"""


EDITOR_PROMPT = """You are the editor of a page listing only the best ANALYTICAL articles about the 2026 United States elections, read by senior policy analysts. The page shows roughly 5 to 10 articles per day across all sources, so be highly selective. You receive each article's title, source and CONTENT (description and opening text). Judge from the content, never from the title alone.

For every CANDIDATE return:
1. type: one of analysis, opinion, explainer, feature, news, roundup, guide, tracker, other. Use news for reporting of events even when it includes some context; guide or tracker for race lists, "races to watch", forecasts and data pages.
2. question: the single specific analytical question the article addresses, written as one short sentence in formal Modern Standard Arabic (at most 20 words). Be specific about the actual subject, e.g. "هل سيدير مشككون في نتائج 2020 انتخابات 2028 في الولايات المتأرجحة؟" rather than "ما السباقات الأهم؟".
3. score (1-10): analytical value to a policy analyst: depth of argument, evidence and data, originality, significance for election outcomes or the direction of US politics, credibility. 9-10 exceptional; 8 strong and clearly worth reading; 6-7 decent but not essential; 5 or below thin. Only analysis, opinion, explainer or feature can score 8 or more. Partisan attack pieces, pieces built on one quote, and celebrity or human-interest angles score 5 or below.
4. match: the id of an EXISTING ISSUE only if the article addresses the same specific analytical question as that issue (compare questions, not headline words). Different angles on the same specific question count as the same issue. Sharing a broad theme ("key races", "polls", "Trump's popularity", "the midterms") is NOT enough. Example: an essay on election-denier candidates for secretary of state and an analysis of the administration's legal push over state voter rolls share the question of federal and partisan pressure on election administration, so they match; that same essay and a list of House races that could decide control do NOT match. When in doubt, null.
5. group: if match is null, give candidates in this batch that address the same specific question the same short label (e.g. "g1"); otherwise null. Same strict rule.
6. reason: at most 15 words in English explaining the score.

Return JSON only, no prose:
{"results":[{"i":0,"type":"analysis","question":"...","score":8,"match":null,"group":null,"reason":"..."}]}
Include one entry for every candidate index."""


VERIFY_PROMPT = """You are an independent second reviewer for a page listing only the best ANALYTICAL articles about the 2026 United States elections, read by senior policy analysts. Another editor has proposed the items below. Your job is to catch mistakes; when in doubt, reject.

For each item you get the article (title, source, content) and, when relevant, a PROPOSED ISSUE (the analytical question of an issue and the title of an article already filed under it).

Return for each item:
- publish_ok: true only if you independently judge the article to be a genuine analytical piece (analysis, opinion, explainer or feature, not a news report, list, guide or tracker), primarily about US elections, with strong analytical value (8 or more out of 10) for a policy analyst.
- same_issue_ok: if a PROPOSED ISSUE is given, true only if the article addresses the same specific analytical question as that issue (a shared broad theme is not enough); if no proposed issue is given, null.

Return JSON only, no prose:
{"results":[{"i":0,"publish_ok":true,"same_issue_ok":null}]}
Include one entry for every item index."""


def ask_claude(client, model, system, blocks, max_tokens=4096):
    """blocks: list of (text, cache) pairs sent as the user message."""
    content = []
    for text, cache in blocks:
        block = {"type": "text", "text": text}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        content.append(block)
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


def title_lines(batch):
    return "\n".join(f"[{i}] {it['source']} | {it['title']}" for i, it in enumerate(batch))


def article_block(i, it):
    return f"[{i}] SOURCE: {it['source']}\nTITLE: {it['title']}\nCONTENT:\n{it['content']}\n"


def triage(client, batch):
    data = ask_claude(client, TRIAGE_MODEL, TRIAGE_PROMPT,
                      [("CANDIDATES:\n" + title_lines(batch), False)], max_tokens=1024)
    if data is None:
        return None
    keep = set()
    for i in data.get("keep", []):
        try:
            keep.add(int(i))
        except (TypeError, ValueError):
            pass
    return keep


def existing_text(stories):
    lines = [f"{s['id']} | {s.get('question') or s['title']}" for s in stories[:MAX_EXISTING]]
    return "EXISTING ISSUES (id | analytical question):\n" + ("\n".join(lines) or "(none)")


def edit(client, batch, stories):
    data = ask_claude(client, MODEL, EDITOR_PROMPT, [
        (existing_text(stories), True),
        ("CANDIDATES:\n\n" + "\n".join(article_block(i, it) for i, it in enumerate(batch)), False),
    ], max_tokens=6000)
    if data is None:
        return None
    return {int(r["i"]): r for r in data.get("results", []) if "i" in r}


def verify(client, entries):
    """entries: list of (item, proposed_issue_or_None) where proposed issue = {question, title}."""
    parts = []
    for i, (it, issue) in enumerate(entries):
        block = article_block(i, it)
        if issue:
            block += f"PROPOSED ISSUE: {issue['question']} | filed article: {issue['title']}\n"
        parts.append(block)
    data = ask_claude(client, MODEL, VERIFY_PROMPT, [("ITEMS:\n\n" + "\n".join(parts), False)],
                      max_tokens=3000)
    if data is None:
        return None
    return {int(r["i"]): r for r in data.get("results", []) if "i" in r}


# ------------------------------------------------------------------ storage

def insert_articles(story_id, arts):
    body = [{
        "story_id": story_id,
        "title": a["title"],
        "source": a["source"],
        "link": a.get("url") or a["link"],
        "published_at": iso(a["published"]),
        "question": a.get("question"),
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
        "link": first.get("url") or first["link"],
        "published_at": iso(first["published"]),
        "story_date": riyadh_date(first["published"]),
        "question": first.get("question"),
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


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def process_batch(client, batch, stories, story_by_id, day_counts):
    """Editor pass, then verification pass. Returns (published, new_issues, filed_under_existing)."""
    proposals = edit(client, batch, stories)
    if proposals is None:
        return None

    passing = []
    for i, it in enumerate(batch):
        r = proposals.get(i) or {}
        it["question"] = (r.get("question") or "").strip() or None
        it["score"] = to_float(r.get("score"))
        if str(r.get("type", "")).lower() not in ANALYTIC_TYPES or it["score"] < MIN_SCORE:
            continue
        if not it["question"]:
            continue
        passing.append((it, r))

    # Proposed placement for each passing article
    entries, plans = [], []
    leaders = {}
    for it, r in passing:
        match = to_int(r.get("match"))
        group = r.get("group")
        if match in story_by_id:
            issue = story_by_id[match]
            entries.append((it, {"question": issue.get("question") or issue["title"], "title": issue["title"]}))
            plans.append(("match", match))
        elif group and str(group) in leaders:
            lead = leaders[str(group)]
            entries.append((it, {"question": lead["question"], "title": lead["title"]}))
            plans.append(("group", str(group)))
        else:
            if group:
                leaders[str(group)] = it
            entries.append((it, None))
            plans.append(("new", str(group) if group else None))

    if not entries:
        return 0, 0, 0
    checks = verify(client, entries)
    if checks is None:
        return None

    published = created = attached = 0
    new_issue_for_group = {}
    for idx, ((it, issue), (kind, ref)) in enumerate(zip(entries, plans)):
        c = checks.get(idx) or {}
        if not c.get("publish_ok"):
            continue
        same = bool(c.get("same_issue_ok"))
        day = riyadh_date(it["published"])

        if kind == "match" and same:
            insert_articles(ref, [it])
            refresh_story(ref)
            attached += 1
            published += 1
            continue
        if kind == "group" and same and ref in new_issue_for_group:
            sid = new_issue_for_group[ref]
            insert_articles(sid, [it])
            refresh_story(sid)
            attached += 1
            published += 1
            continue

        # New issue (also the fallback when the second reviewer rejects a placement)
        if day_counts.get(day, 0) >= DAILY_CAP and it["score"] < 9:
            continue
        row = create_story([it])
        created += 1
        published += 1
        day_counts[row["story_date"]] = day_counts.get(row["story_date"], 0) + 1
        info = {"id": row["id"], "title": row["title"], "question": it["question"]}
        stories.insert(0, info)
        story_by_id[row["id"]] = info
        if kind == "new" and ref:
            new_issue_for_group[ref] = row["id"]
    return published, created, attached


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
    log(f"Models: triage {TRIAGE_MODEL}, editor/verifier {MODEL} | min score {MIN_SCORE}, "
        f"daily cap {DAILY_CAP} | window starts {iso(since)} | link decoder "
        f"{'ready' if _gdecode else 'MISSING'}")

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

    # Triage by title (parallel, cheap model)
    batches = [items[b:b + TRIAGE_BATCH] for b in range(0, len(items), TRIAGE_BATCH)]
    shortlisted = []
    with ThreadPoolExecutor(max_workers=TRIAGE_WORKERS) as pool:
        for batch, keep in zip(batches, pool.map(lambda bt: triage(client, bt), batches)):
            if keep is None:
                continue
            shortlisted.extend(it for i, it in enumerate(batch) if i in keep)
            mark_seen([it["key"] for i, it in enumerate(batch) if i not in keep])
    log(f"Shortlisted as possible analysis: {len(shortlisted)}")

    # Content: no readable content, no publishing
    readable = load_content(shortlisted)
    log(f"Readable content: {len(readable)} | no content (not published, retried next run): "
        f"{len(shortlisted) - len(readable)}")

    stories = sb_all("election_stories", {
        "select": "id,title,question,updated_at",
        "updated_at": f"gte.{iso(now - timedelta(days=MATCH_DAYS))}",
        "order": "updated_at.desc",
    })
    story_by_id = {s["id"]: s for s in stories}
    day_counts = {}
    for r in sb_all("election_stories", {"select": "story_date",
                                         "story_date": f"gte.{riyadh_date(since)}"}):
        day_counts[r["story_date"]] = day_counts.get(r["story_date"], 0) + 1
    log(f"Existing issues in the last {MATCH_DAYS} days: {len(stories)}")

    totals = [0, 0, 0]
    for b in range(0, len(readable), BATCH_SIZE):
        if datetime.now(UTC) > deadline:
            log("Time budget reached; the rest waits for the next run")
            break
        batch = readable[b:b + BATCH_SIZE]
        res = process_batch(client, batch, stories, story_by_id, day_counts)
        if res is None:
            log(f"Batch {b // BATCH_SIZE + 1}: skipped, will retry next run")
            continue
        totals = [x + y for x, y in zip(totals, res)]
        mark_seen([it["key"] for it in batch])
        log(f"Batch {b // BATCH_SIZE + 1}: {len(batch)} read | published {res[0]}")

    log(f"Done. Published {totals[0]} | new issues {totals[1]} | added to existing issues {totals[2]}")


if __name__ == "__main__":
    main()
