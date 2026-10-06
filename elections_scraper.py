#!/usr/bin/env python3
"""
Ayn - US Elections 2026 monitor. Three categories:
  updates : أبرز المستجدات الانتخابية  - key analysis, developments, statements, positions and
            campaign promises, with priority for oil, energy and the region's economy.
  senate  : المرشحون لمجلس الشيوخ     - articles directly about the tracked swing-state candidates.
  house   : مجلس النواب              - notable statements by House members on Iran.

Every run (every 4 hours):
  1. Search the approved sources via Google News RSS and the sources' own RSS feeds
     (election terms, election + energy, candidate names, House + Iran).
  2. Triage by title (Claude Haiku).
  3. Final selection (Claude Sonnet): category, score, person, and issue grouping.
  4. Save to Supabase. Each issue's count grows as more articles cover it.

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
CATEGORIES = ("updates", "senate", "house")
BATCH_SIZE = 20          # candidates per final-selection call
TRIAGE_BATCH = 50        # titles per triage call
TRIAGE_WORKERS = 6       # parallel triage calls
TIME_BUDGET_MIN = 45     # stop AI work after this many minutes; the rest waits for the next run
MATCH_DAYS = 7           # compare new items with issues from the last 7 days
MAX_EXISTING = 400       # max existing issues sent to Claude per call
MAX_CANDIDATES = 10000   # safety cap per run
GOOGLE_PAUSE = 1.0       # seconds between Google News requests
GOOGLE_QUERY_LEN = 80    # short OR-groups: long queries make Google ignore site:
SKIP_TITLE_RE = re.compile(
    r"^\s*(watch|video|live|listen|photos?)\s*[:|]|live updates|week in politics|"
    r"in brief|toplines|cross-tabs|voter guide|countdown|newsletter|podcast", re.I)
HOUSE_RE = re.compile(r"\biran", re.I)
HOUSE_CTX_RE = re.compile(r"\b(rep\.|congress\w*|lawmakers?|house)\b", re.I)


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

def candidates_text(cands):
    return "\n".join(f"- {c['name']} ({c['state']}, {c['party']})" for c in cands)


def triage_prompt(cands):
    return f"""You triage news items for a monitoring page about the 2026 United States elections, read by senior policy analysts in the Gulf region. The page has three categories:

A. ELECTION UPDATES: analytical articles and notable developments, statements, positions and campaign promises that are directly about US elections (2026 midterms, races, campaigns, voters, polling trends, voting rules, redistricting, the 2028 race). Priority: oil, energy, gas prices, sanctions, Iran and Middle East policy, and issues that could affect the Gulf region's economy.
B. SENATE CANDIDATES: articles directly about one of these tracked candidates (their campaign, statements, positions, promises, debates, ads, polls on their race, controversies):
{candidates_text(cands)}
C. HOUSE ON IRAN: notable statements or positions by members of the US House of Representatives about Iran (the war, negotiations, sanctions, war powers, oil).

Keep an item if it could plausibly belong to A, B or C. Drop: items not about these subjects, items that only mention elections or a candidate in passing, videos, live blogs, podcasts, newsletters, roundups and digests, and non-English items. When unsure about a relevant item, keep it; a stricter editor reviews next.

Return JSON only: {{"keep":[0,3,7]}}"""


def editor_prompt(cands):
    return f"""You are the final editor of a monitoring page about the 2026 United States elections, read by senior policy analysts in the Gulf region. Precision matters more than volume. Assign each candidate to at most one category, or reject it.

CATEGORIES
- "senate": the article is directly about one of these tracked Senate candidates (their campaign, statements, positions, promises, debates, ads, fundraising, polls on their race, controversies). A passing mention is not enough.
{candidates_text(cands)}
- "house": the article reports a notable statement or position by a member of the US House of Representatives about Iran (the war, negotiations, sanctions, war powers, oil). Senators do not belong here.
- "updates": strictly about US elections. Either a strong analytical piece, or a notable development, statement, position or campaign promise with clear significance for the elections. Give priority to oil, energy, gas prices, sanctions, Iran and Middle East policy, and issues that could affect the Gulf region's economy. Routine campaign news, horse-race chatter, roundups, digests, race lists, poll toplines, partisan attack lines and celebrity angles do not qualify.
If an item fits "senate" or "house", prefer that category over "updates".

For every CANDIDATE return:
1. category: "updates", "senate", "house", or null to reject.
2. score (1-10): value to a policy analyst. 9-10 essential; 8 strong; 7 useful and clearly relevant; 6 or below marginal. Items on oil, energy or the Gulf economy that are directly tied to the elections deserve extra weight.
3. person: for "senate", the candidate's full name exactly as written in the list above; for "house", the House member's full name (without "Rep."); otherwise null.
4. state and party: for "house" only, the member's state (full name) and party ("Republican", "Democrat" or "Independent"); otherwise null.
5. match: the id of an EXISTING ISSUE in the SAME category only if the item covers the same specific story or question (for senate, also the same candidate). Sharing a word or a broad theme is not enough. If the title does not make it clear, null.
6. group: if match is null, the same short label (e.g. "g1") for candidates in this batch that cover the same specific story in the same category; otherwise null.

Judge from the title, source and snippet. A note "[found via candidate search]" means the article text mentions a tracked candidate somewhere; decide from the title whether it is directly about them. Return JSON only, no prose:
{{"results":[{{"i":0,"category":"senate","score":8,"person":"Ken Paxton","state":null,"party":null,"match":null,"group":null}}]}}
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
        if it.get("via") == "senate":
            line += " [found via candidate search]"
        lines.append(line)
    return "\n".join(lines)


def triage(client, system, batch):
    data = ask_claude(client, TRIAGE_MODEL, system,
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


def judge(client, system, batch, stories):
    existing = "\n".join(f"{s['id']} | {s.get('category') or 'updates'} | {s['title']}"
                         for s in stories[:MAX_EXISTING]) or "(none)"
    content = [
        {"type": "text", "text": f"EXISTING ISSUES (id | category | title of first article):\n{existing}",
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"CANDIDATES:\n{candidate_lines(batch)}"},
    ]
    data = ask_claude(client, MODEL, system, content, max_tokens=5000)
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


def create_story(arts, meta):
    arts = sorted(arts, key=lambda a: a["published"])
    first = arts[0]
    row = sb("POST", "election_stories", body={
        "title": first["title"],
        "source": first["source"],
        "link": first["link"],
        "published_at": iso(first["published"]),
        "story_date": riyadh_date(first["published"]),
        "source_count": len(arts),
        "category": meta["category"],
        "person": meta.get("person"),
        "state": meta.get("state"),
        "party": meta.get("party"),
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


def clean(v):
    v = (v or "").strip() if isinstance(v, str) else ""
    return v or None


def main():
    kw = load_json("keywords.json")
    sources = load_json("sources.json")
    src_idx = build_source_index(sources)
    cands = kw.get("senate_candidates", [])
    cand_by_name = {c["name"].lower(): c for c in cands}
    rules = {
        "updates": (float(kw.get("min_score", 8)), int(kw.get("daily_cap", 10))),
        "senate": (float(kw.get("senate_min_score", 7)), int(kw.get("senate_daily_cap", 40))),
        "house": (float(kw.get("house_min_score", 7)), int(kw.get("house_daily_cap", 15))),
    }

    terms = [t for t in kw.get("search_terms", []) if t.strip()]
    names = [c["name"] for c in cands] + [t for t in kw.get("names", []) if t.strip()]
    election_re = keyword_regex(terms)
    names_re = keyword_regex(names) if names else None

    queries = [(q, "updates", "all") for q in term_chunks(terms, max_len=GOOGLE_QUERY_LEN)]
    if kw.get("energy_query"):
        queries.append((kw["energy_query"], "updates", "all"))
    if names:
        queries += [(q, "senate", "media") for q in term_chunks(names, max_len=GOOGLE_QUERY_LEN)]
    if kw.get("house_query"):
        queries.append((kw["house_query"], "house", "media"))

    now = datetime.now(UTC)
    backfill = datetime.fromisoformat(kw["backfill_start"]).replace(tzinfo=RIYADH).astimezone(UTC)
    since = max(backfill, now - timedelta(days=int(kw.get("lookback_days", 5))))
    days = max(1, int((now - since).total_seconds() // 86400) + 1)
    log(f"Models: triage {TRIAGE_MODEL}, final {MODEL} | rules {rules} | window starts {iso(since)}")

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
                keep = []
                for it in items:
                    text = f"{it['title']} {it['snippet']}"
                    if names_re and names_re.search(text):
                        it["via"] = "senate"
                    elif HOUSE_RE.search(text) and HOUSE_CTX_RE.search(text):
                        it["via"] = "house"
                    elif election_re.search(text):
                        it["via"] = "updates"
                    else:
                        continue
                    keep.append(it)
                raw.extend(keep)
                log(f"RSS  {s['name']}: {len(keep)} matching")
            except Exception as ex:  # noqa: BLE001
                log(f"RSS  {s['name']} failed: {ex}")

    for s in sources:
        is_media = s.get("type", "media") == "media"
        for domain in s["domains"]:
            got = 0
            for q, via, scope in queries:
                if scope == "media" and not is_media:
                    continue
                try:
                    items = fetch_google(domain, q, days)
                    for it in items:
                        it["via"] = via
                    raw.extend(items)
                    got += len(items)
                except Exception as ex:  # noqa: BLE001
                    log(f"GNEWS {domain} failed: {ex}")
                time.sleep(GOOGLE_PAUSE)
            log(f"GNEWS {domain}: {got} results")

    candidates = {}
    priority = {"senate": 3, "house": 2, "updates": 1}
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
        if prev is None:
            candidates[key] = it
            continue
        if priority.get(it.get("via"), 0) > priority.get(prev.get("via"), 0):
            prev["via"] = it["via"]
        if prev["origin"] == "google" and it["origin"] != "google":
            it["via"] = prev["via"]
            candidates[key] = it

    log(f"Fetched {len(raw)} raw items | dropped: {drops}")
    items = sorted(candidates.values(), key=lambda x: x["published"])[:MAX_CANDIDATES]
    log(f"New candidates: {len(items)}")
    if not items:
        log("Nothing new.")
        return

    client = anthropic.Anthropic()
    deadline = START + timedelta(minutes=TIME_BUDGET_MIN)
    t_prompt, e_prompt = triage_prompt(cands), editor_prompt(cands)

    batches = [items[b:b + TRIAGE_BATCH] for b in range(0, len(items), TRIAGE_BATCH)]
    shortlisted = []
    with ThreadPoolExecutor(max_workers=TRIAGE_WORKERS) as pool:
        for batch, keep in zip(batches, pool.map(lambda bt: triage(client, t_prompt, bt), batches)):
            if keep is None:
                continue
            shortlisted.extend(it for i, it in enumerate(batch) if i in keep)
            mark_seen([it["key"] for i, it in enumerate(batch) if i not in keep])
    log(f"Shortlisted: {len(shortlisted)}")

    stories = sb_all("election_stories", {
        "select": "id,title,category,updated_at",
        "updated_at": f"gte.{iso(now - timedelta(days=MATCH_DAYS))}",
        "order": "updated_at.desc",
    })
    story_cat = {s["id"]: (s.get("category") or "updates") for s in stories}
    day_counts = {}
    for r in sb_all("election_stories", {"select": "story_date,category",
                                         "story_date": f"gte.{riyadh_date(since)}"}):
        k = (r.get("category") or "updates", r["story_date"])
        day_counts[k] = day_counts.get(k, 0) + 1
    log(f"Existing issues in the last {MATCH_DAYS} days: {len(stories)}")

    totals = {c: [0, 0] for c in CATEGORIES}   # [new issues, added to existing]
    for b in range(0, len(shortlisted), BATCH_SIZE):
        if datetime.now(UTC) > deadline:
            log("Time budget reached; remaining shortlisted items wait for the next run")
            break
        batch = shortlisted[b:b + BATCH_SIZE]
        results = judge(client, e_prompt, batch, stories)
        if results is None:
            log(f"Batch {b // BATCH_SIZE + 1}: skipped, will retry next run")
            continue

        groups, singles, matches = {}, [], {}
        for i, it in enumerate(batch):
            r = results.get(i) or {}
            cat = r.get("category")
            if cat not in CATEGORIES:
                continue
            score = to_float(r.get("score"))
            min_score, cap = rules[cat]
            if score < min_score:
                continue
            meta = {"category": cat}
            if cat == "senate":
                c = cand_by_name.get((clean(r.get("person")) or "").lower())
                if not c:
                    continue
                meta.update(person=c["name"], state=c["state"], party=c["party"])
            elif cat == "house":
                person = clean(r.get("person"))
                if not person:
                    continue
                meta.update(person=person, state=clean(r.get("state")), party=clean(r.get("party")))
            it["score"] = score

            match = to_int(r.get("match"))
            if match in story_cat and story_cat[match] == cat:
                matches.setdefault(match, []).append(it)
                continue
            day = riyadh_date(it["published"])
            if day_counts.get((cat, day), 0) >= cap and (cat != "updates" or score < 9):
                continue
            if r.get("group"):
                g = groups.setdefault(f"{cat}|{r['group']}|{meta.get('person')}", {"meta": meta, "arts": []})
                g["arts"].append(it)
            else:
                singles.append({"meta": meta, "arts": [it]})

        for sid, arts in matches.items():
            insert_articles(sid, arts)
            refresh_story(sid)
            totals[story_cat[sid]][1] += len(arts)

        for g in list(groups.values()) + singles:
            row = create_story(g["arts"], g["meta"])
            cat = g["meta"]["category"]
            totals[cat][0] += 1
            story_cat[row["id"]] = cat
            stories.insert(0, {"id": row["id"], "title": row["title"], "category": cat})
            k = (cat, row["story_date"])
            day_counts[k] = day_counts.get(k, 0) + 1

        mark_seen([it["key"] for it in batch])
        log(f"Batch {b // BATCH_SIZE + 1}: {len(batch)} reviewed")

    log("Done. " + " | ".join(f"{c}: {n} new issues, {a} added to existing" for c, (n, a) in totals.items()))


if __name__ == "__main__":
    main()
