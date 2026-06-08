#!/usr/bin/env python3
"""
arxiv_daily.py — 每日 arXiv 论文推荐引擎。

抓最近 arXiv 新论文 → 按与"兴趣种子"的相关度排序 → (可选)LLM 写 TL;DR → 输出 markdown / 发邮件。

兴趣种子三选一(可叠加)：
  --seed-author "First Last"   用某研究者的 arXiv 论文(标题+摘要)当兴趣画像
  --notes /path/to/md_dir      读一个文件夹里的 .md 笔记当兴趣画像
  --keywords "LLM inference, quantization, ..."   关键词兜底

排序：纯 Python TF-IDF 余弦（无需安装任何东西）。

环境变量(可选)：
  LLM_API_KEY / LLM_BASE_URL / LLM_MODEL   设了就让 LLM 写 TL;DR
  SMTP_SENDER_EMAIL / SMTP_APP_PASSWORD / DIGEST_RECIPIENT / SMTP_HOST / SMTP_PORT
                                            设了就发邮件，否则写文件

用法:
  python3 arxiv_daily.py --seed-author "First Last" --cats cs.CL,cs.LG,cs.AI --top 15 -o digest.md
"""
import argparse, glob, json, math, os, re, sys, urllib.parse, urllib.request
from collections import Counter
from xml.etree import ElementTree as ET

UA = "arxiv-daily/1.0"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
STOP = set(("the a an of and for to in on with via using under over into from is are be we our this that "
            "as at by or not no all any can may will using based toward towards new using approach method "
            "results show paper propose present study work models model task tasks data use used between "
            "their these those it its which when while where than then also more most such these").split())


def _load_secrets(path):
    """本地运行时把 .secrets.env 读进环境变量（不覆盖已存在的）。GitHub Actions 用 repo secrets。"""
    try:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k, v = k.strip(), v.split("#", 1)[0].strip()
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except FileNotFoundError:
        pass


_load_secrets(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".secrets.env"))


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30).read()


def _arxiv_query(search_query, max_results, sort="submittedDate"):
    url = (f"http://export.arxiv.org/api/query?search_query={search_query}"
           f"&sortBy={sort}&sortOrder=descending&max_results={max_results}")
    try:
        root = ET.fromstring(_get(url))
    except Exception as e:
        print(f"[warn] arxiv query failed: {e}", file=sys.stderr)
        return []
    out = []
    for e in root.findall("a:entry", NS):
        title = re.sub(r"\s+", " ", (e.findtext("a:title", "", NS) or "").strip())
        summ = re.sub(r"\s+", " ", (e.findtext("a:summary", "", NS) or "").strip())
        pub = e.findtext("a:published", "", NS)[:10]
        aid = (e.findtext("a:id", "", NS) or "").rsplit("/", 1)[-1]
        authors = [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)]
        pc = e.find("arxiv:primary_category", NS)
        cat = pc.get("term") if pc is not None else ""
        out.append({"title": title, "abstract": summ, "date": pub, "id": aid,
                    "link": f"https://arxiv.org/pdf/{aid}", "authors": authors, "cat": cat})
    return out


def fetch_candidates(cats, pool):
    q = "+OR+".join(f"cat:{c.strip()}" for c in cats)
    return _arxiv_query(urllib.parse.quote_plus("(") + q.replace("+OR+", "+OR+") + urllib.parse.quote_plus(")")
                        if False else q, pool)


def seed_text(author, notes_dir, keywords):
    chunks = []
    if author:
        docs = _arxiv_query(urllib.parse.quote(f'au:"{author}"'), 40)
        ml = [d for d in docs if d["cat"].startswith(("cs.", "stat.", "eess."))] or docs
        chunks += [f"{d['title']}. {d['abstract']}" for d in ml]
        print(f"[seed] author '{author}': {len(ml)} arXiv papers", file=sys.stderr)
    if notes_dir and os.path.isdir(notes_dir):
        files = glob.glob(os.path.join(notes_dir, "**/*.md"), recursive=True)
        for f in files[:500]:
            try:
                chunks.append(open(f, errors="ignore").read())
            except Exception:
                pass
        print(f"[seed] notes '{notes_dir}': {len(files)} md files", file=sys.stderr)
    if keywords:
        chunks.append(keywords.replace(",", " ") * 3)  # 关键词加权
    return "\n".join(chunks)


def tokenize(t):
    return [w for w in re.findall(r"[a-z][a-z\-]{2,}", (t or "").lower()) if w not in STOP]


def rank(seed, candidates, liked_texts=None, disliked_texts=None, alpha=1.0, beta=0.8, gamma=0.6):
    """Rocchio 相关性反馈：q = α·种子 + β·均值(👍) − γ·均值(👎)，按 cos(候选, q) 排序。"""
    liked_texts = liked_texts or []
    disliked_texts = disliked_texts or []
    cand_docs = [tokenize(c["title"] + " " + c["abstract"]) for c in candidates]
    seed_doc = tokenize(seed)
    liked_docs = [tokenize(t) for t in liked_texts]
    disliked_docs = [tokenize(t) for t in disliked_texts]
    corpus = [seed_doc] + cand_docs + liked_docs + disliked_docs
    if not any(corpus):
        return [(0.0, c) for c in candidates]
    df = Counter()
    for d in corpus:
        for w in set(d):
            df[w] += 1
    N = max(1, len(corpus))
    idf = {w: math.log(N / (1 + c)) + 1 for w, c in df.items()}

    def vec(toks):
        if not toks:
            return {}
        tf = Counter(toks)
        return {w: (f / len(toks)) * idf.get(w, 0.0) for w, f in tf.items()}

    def centroid(vs):
        vs = [v for v in vs if v]
        if not vs:
            return {}
        c = {}
        for v in vs:
            for w, x in v.items():
                c[w] = c.get(w, 0.0) + x
        return {w: x / len(vs) for w, x in c.items()}

    def cos(a, b):
        if not a or not b:
            return 0.0
        dot = sum(x * b.get(w, 0.0) for w, x in a.items())
        na = math.sqrt(sum(x * x for x in a.values()))
        nb = math.sqrt(sum(x * x for x in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def unit(v):
        n = math.sqrt(sum(x * x for x in v.values()))
        return {w: x / n for w, x in v.items()} if n else {}

    def add(q, v, weight):
        for w, x in v.items():
            q[w] = q.get(w, 0.0) + weight * x

    # 每部分先单位化再加权：否则 27 篇拼成的巨型种子、或单篇的稀有词会主导方向
    q = {}
    add(q, unit(vec(seed_doc)), alpha)
    add(q, unit(centroid([unit(vec(d)) for d in liked_docs])), beta)
    add(q, unit(centroid([unit(vec(d)) for d in disliked_docs])), -gamma)
    scored = [(cos(q, vec(cand_docs[i])), candidates[i]) for i in range(len(candidates))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def load_feedback(path):
    """读 feedback.jsonl（每行 {id, vote: up|down, ts}）。同一篇取最新一票。"""
    latest = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                bid = re.sub(r"v\d+$", "", o.get("id", ""))
                if bid and o.get("vote") in ("up", "down"):
                    latest[bid] = (o["vote"], o.get("id", bid))
    except FileNotFoundError:
        pass
    liked = {i for v, i in latest.values() if v == "up"}
    disliked = {i for v, i in latest.values() if v == "down"}
    return liked, disliked, set(latest.keys())


def fetch_by_ids(ids):
    """按 arXiv id 批量取 title+abstract（给已投票论文做向量）。"""
    ids = [i for i in ids if i]
    out = {}
    for j in range(0, len(ids), 50):
        chunk = ids[j:j + 50]
        url = f"http://export.arxiv.org/api/query?id_list={','.join(chunk)}&max_results={len(chunk)}"
        try:
            root = ET.fromstring(_get(url))
        except Exception as e:
            print(f"[warn] fetch_by_ids failed: {e}", file=sys.stderr)
            continue
        for e in root.findall("a:entry", NS):
            title = re.sub(r"\s+", " ", (e.findtext("a:title", "", NS) or "").strip())
            summ = re.sub(r"\s+", " ", (e.findtext("a:summary", "", NS) or "").strip())
            out[(e.findtext("a:id", "", NS) or "").rsplit("/", 1)[-1]] = f"{title} {summ}"
    return out


def _translate_prompt(abstract):
    return ("把下面这段论文摘要翻译成简洁通顺的中文，只输出译文本身，不要加任何前言、标题或解释：\n\n"
            + abstract)


def _anthropic_text(prompt, default_model="claude-haiku-4-5", max_tokens=200):
    import anthropic  # 官方 SDK
    client = anthropic.Anthropic(api_key=os.environ["LLM_API_KEY"])
    model = os.environ.get("LLM_MODEL", default_model)
    msg = client.messages.create(model=model, max_tokens=max_tokens,
                                 messages=[{"role": "user", "content": prompt}])
    return next((b.text for b in msg.content if b.type == "text"), "").strip()


def _openai_text(prompt, default_model="gpt-4o-mini", max_tokens=200):
    key = os.environ["LLM_API_KEY"]
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com").rstrip("/")
    model = os.environ.get("LLM_MODEL", default_model)
    body = json.dumps({"model": model, "temperature": 0.3, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": UA})
    return json.loads(urllib.request.urlopen(req, timeout=40).read())["choices"][0]["message"]["content"].strip()


def _llm_text(prompt, **kw):
    """anthropic → 官方 SDK；其余（deepseek/siliconflow/openai）→ OpenAI 兼容 HTTP。"""
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return None
    provider = (os.environ.get("LLM_PROVIDER") or "").lower()
    if provider == "anthropic" or key.startswith("sk-ant"):
        return _anthropic_text(prompt, **kw)
    return _openai_text(prompt, **kw)


def llm_translate(abstract):
    try:
        return _llm_text(_translate_prompt(abstract), max_tokens=1024)
    except Exception as e:
        print(f"[warn] llm translate failed: {e}", file=sys.stderr)
        return None


def llm_authors(arxiv_id, fallback):
    """从 arXiv HTML 首页用 LLM 抽取『作者（机构）』；失败回退到 arXiv 作者名。"""
    try:
        html = _get(f"https://arxiv.org/html/{arxiv_id}").decode("utf-8", "ignore")
        t = re.sub(r"<(script|style).*?</\1>", " ", html, flags=re.S)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"\s+", " ", t).strip()[:5000]
        out = _llm_text(
            "下面是一篇 arXiv 论文首页的文本。只提取所有作者及其所属机构，用中文按『作者（机构）』格式、"
            "逗号分隔列在一行；某作者机构缺失就只写其姓名。不要输出论文标题、不要解释、不要任何多余内容：\n\n" + t,
            max_tokens=300)
        if out and out.strip():
            return out.strip()
    except Exception as e:
        print(f"[warn] authors extract failed ({arxiv_id}): {e}", file=sys.stderr)
    return ", ".join(fallback) if fallback else ""


def render(items, seed_desc):
    import time
    L = [f"# arxiv daily paper · {time.strftime('%Y-%m-%d')}", "_按相关度排序_\n"]
    for it in items:
        c = it["c"]
        L.append(f"### {it['i']}. {c['title']}")
        L.append(f"_{c['date']} · {c['cat']} · 相关度 {it['score']:.3f}_")
        if it["authors"]:
            L.append(f"**作者**：{it['authors']}")
        L.append(f"— [{c['id']} (pdf)]({c['link']})")
        L.append("")
        if it["zh"]:
            L.append("**中文摘要**：")
            L.append(it["zh"])
            L.append("")
    L.append(f"\n---\n_共 {len(items)} 篇_")
    return "\n".join(L)


def send_email(subject, html_or_text):
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    sender = os.environ.get("SMTP_SENDER_EMAIL")
    pw = os.environ.get("SMTP_APP_PASSWORD")
    to = os.environ.get("DIGEST_RECIPIENT")
    if not (sender and pw and to):
        return False
    import smtplib
    from email.mime.text import MIMEText
    msg = MIMEText(html_or_text, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(sender, pw)
        s.sendmail(sender, [to], msg.as_string())
    return True


def _tg(method, payload, tok):
    req = urllib.request.Request(f"https://api.telegram.org/bot{tok}/{method}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def telegram_ingest(tok, fb_path, offset_path):
    """把 getUpdates 里的 👍/👎 按钮点击写进 feedback.jsonl，并推进 offset。"""
    off = 0
    try:
        off = int(open(offset_path).read().strip())
    except Exception:
        pass
    try:
        d = _tg("getUpdates", {"offset": off, "timeout": 0, "allowed_updates": ["callback_query"]}, tok)
    except Exception as e:
        print(f"[warn] tg getUpdates: {e}", file=sys.stderr)
        return 0
    n, last = 0, off
    with open(fb_path, "a") as fb:
        for u in d.get("result", []):
            last = max(last, u["update_id"] + 1)
            cq = u.get("callback_query")
            if not cq:
                continue
            v, _, pid = cq.get("data", "").partition(":")
            vote = {"up": "up", "dn": "down"}.get(v)
            if vote and pid:
                fb.write(json.dumps({"id": pid, "vote": vote}) + "\n")
                n += 1
                try:
                    _tg("answerCallbackQuery", {"callback_query_id": cq["id"],
                        "text": "👍 已记录，会多推这类" if vote == "up" else "👎 已记录，会少推这类"}, tok)
                except Exception:
                    pass
    if last != off:
        open(offset_path, "w").write(str(last))
    if n:
        print(f"[info] ingested {n} telegram votes", file=sys.stderr)
    return n


def enrich(scored, top):
    """每篇只算一次：中文翻译 + 作者(机构)。供 Telegram 和邮件复用，避免重复调 LLM。"""
    items = []
    for i, (score, c) in enumerate(scored[:top], 1):
        items.append({"i": i, "score": score, "c": c,
                      "zh": llm_translate(c["abstract"]),
                      "authors": llm_authors(c["id"], c.get("authors"))})
    return items


def telegram_send(tok, chat_id, items):
    import time
    try:
        _tg("sendMessage", {"chat_id": chat_id,
            "text": f"📚 arxiv daily paper · {time.strftime('%Y-%m-%d')} · 共 {len(items)} 篇（点 👍/👎 调教推荐）"}, tok)
    except Exception as e:
        print(f"[warn] tg head: {e}", file=sys.stderr)
    sent = 0
    for it in items:
        c = it["c"]
        parts = [f"{it['i']}. {c['title']}", f"{c['date']} · {c['cat']} · 相关度 {it['score']:.3f}"]
        if it["authors"]:
            parts.append(f"作者：{it['authors']}")
        parts.append(c["link"])
        if it["zh"]:
            parts.append(f"\n【中文摘要】\n{it['zh']}")
        text = "\n".join(parts)
        if len(text) > 4000:
            text = text[:3980] + " …"
        kb = {"inline_keyboard": [[{"text": "👍 喜欢", "callback_data": f"up:{c['id']}"},
                                   {"text": "👎 不要", "callback_data": f"dn:{c['id']}"}]]}
        try:
            _tg("sendMessage", {"chat_id": chat_id, "text": text, "reply_markup": kb,
                                "disable_web_page_preview": True}, tok)
            sent += 1
        except Exception as e:
            print(f"[warn] tg send {c['id']}: {e}", file=sys.stderr)
    return sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-author", default=os.environ.get("SEED_AUTHOR"))
    ap.add_argument("--notes", default=os.environ.get("MD_NOTES_PATH"))
    ap.add_argument("--keywords", default=os.environ.get("COLD_START_KEYWORDS"))
    ap.add_argument("--cats", default=os.environ.get("ARXIV_CATEGORIES", "cs.CL,cs.LG,cs.AI").replace(" ", ","))
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--top", type=int, default=int(os.environ.get("PAPERS_PER_DAY", "15")))
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--feedback", default=os.environ.get("FEEDBACK_FILE"))
    a = ap.parse_args()

    cats = [c for c in a.cats.split(",") if c]
    seed = seed_text(a.seed_author, a.notes, a.keywords)
    seed_desc = a.seed_author or a.notes or (a.keywords or "")[:50]
    here = os.path.dirname(os.path.abspath(__file__))
    fb_path = a.feedback or os.path.join(here, "feedback.jsonl")
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if tok:
        telegram_ingest(tok, fb_path, os.path.join(here, "tg_offset.txt"))  # 先收上一轮的 👍/👎
    cands = fetch_candidates(cats, a.pool)
    print(f"[info] fetched {len(cands)} candidates from {cats}", file=sys.stderr)
    liked_ids, disliked_ids, voted_bases = load_feedback(fb_path)
    if voted_bases:
        cands = [c for c in cands if re.sub(r"v\d+$", "", c["id"]) not in voted_bases]
        print(f"[info] feedback 👍{len(liked_ids)} 👎{len(disliked_ids)}; {len(cands)} left after excluding voted", file=sys.stderr)
    liked_texts = list(fetch_by_ids(liked_ids).values())
    disliked_texts = list(fetch_by_ids(disliked_ids).values())
    scored = rank(seed, cands, liked_texts, disliked_texts)
    items = enrich(scored, a.top)   # 翻译+作者只算一次

    if tok and chat:
        print(f"[ok] telegram sent {telegram_send(tok, chat, items)}")
    md = render(items, seed_desc)
    if os.environ.get("SMTP_SENDER_EMAIL") and send_email("arxiv daily paper", md):
        print("[ok] emailed digest")
    if a.out:
        open(a.out, "w").write(md)
        print(f"[ok] written: {a.out}")
    elif not (tok and chat) and not os.environ.get("SMTP_SENDER_EMAIL"):
        print(md)


if __name__ == "__main__":
    main()
