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


def rank(seed, candidates):
    if not seed.strip():
        return [(0.0, c) for c in candidates]
    docs = [tokenize(seed)] + [tokenize(c["title"] + " " + c["abstract"]) for c in candidates]
    df = Counter()
    for d in docs:
        for w in set(d):
            df[w] += 1
    N = len(docs)
    idf = {w: math.log(N / (1 + c)) + 1 for w, c in df.items()}

    def vec(toks):
        tf = Counter(toks)
        return {w: (f / len(toks)) * idf[w] for w, f in tf.items()} if toks else {}

    def cos(a, b):
        if not a or not b:
            return 0.0
        dot = sum(a[w] * b.get(w, 0) for w in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    sv = vec(docs[0])
    scored = [(cos(sv, vec(docs[i + 1])), candidates[i]) for i in range(len(candidates))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


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


def render(scored, top, seed_desc):
    import time
    L = [f"# arxiv daily paper · {time.strftime('%Y-%m-%d')}", "_按相关度排序_\n"]
    for i, (score, c) in enumerate(scored[:top], 1):
        zh = llm_translate(c["abstract"])
        authors = llm_authors(c["id"], c.get("authors"))
        L.append(f"### {i}. {c['title']}")
        L.append(f"_{c['date']} · {c['cat']} · 相关度 {score:.3f}_")
        if authors:
            L.append(f"**作者**：{authors}")
        L.append(f"— [{c['id']} (pdf)]({c['link']})")
        L.append("")
        if zh:
            L.append("**中文**：")
            L.append(zh)
            L.append("")
        L.append("**English**:")
        L.append(c["abstract"])
        L.append("")
    L.append(f"\n---\n_arxiv_daily.py · 候选池 {len(scored)} 篇 → 取 top {top}_")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-author", default=os.environ.get("SEED_AUTHOR"))
    ap.add_argument("--notes", default=os.environ.get("MD_NOTES_PATH"))
    ap.add_argument("--keywords", default=os.environ.get("COLD_START_KEYWORDS"))
    ap.add_argument("--cats", default=os.environ.get("ARXIV_CATEGORIES", "cs.CL,cs.LG,cs.AI").replace(" ", ","))
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--top", type=int, default=int(os.environ.get("PAPERS_PER_DAY", "15")))
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    cats = [c for c in a.cats.split(",") if c]
    seed = seed_text(a.seed_author, a.notes, a.keywords)
    seed_desc = a.seed_author or a.notes or (a.keywords or "")[:50]
    cands = fetch_candidates(cats, a.pool)
    print(f"[info] fetched {len(cands)} candidates from {cats}", file=sys.stderr)
    scored = rank(seed, cands)
    md = render(scored, a.top, seed_desc)

    if send_email("arxiv daily paper", md):
        print("[ok] emailed digest")
    if a.out:
        open(a.out, "w").write(md)
        print(f"[ok] written: {a.out}")
    elif not os.environ.get("SMTP_SENDER_EMAIL"):
        print(md)


if __name__ == "__main__":
    main()
