# daily-arxiv-agent

每天抓最新 arXiv 论文 → 按相关度排序 → Claude (Opus 4.8) 翻译中文摘要 → **邮件推送** + **建一个 GitHub issue（每篇一条评论，对评论点 👍/👎 reaction 来反馈，调整后续推荐）**。

- **种子**：聚焦高效 LLM 推理方向的论文（低比特量化 / KV-cache / 稀疏注意力 / 投机解码）。种子作者放在 Actions Secret `SEED_AUTHOR`，不写进仓库。
- **排序**：纯 Python TF-IDF 余弦 + Rocchio 相关性反馈（`q = 种子 + 👍均值 − 👎均值`），已投票的论文不再出现。零额外依赖。
- **摘要**：Anthropic **Opus 4.8**（官方 SDK）。可用 `LLM_MODEL` 覆盖。
- **投递**：每天发**邮件**（SMTP）；同时在本仓库 **Issues** 建一条当天的 digest issue，每篇论文一条评论。
- **反馈闭环（零服务器、无需第三方 token）**：对每条评论点 GitHub 原生 👍/👎 reaction → 第二天的 Action 用 GitHub API（自带 `GITHUB_TOKEN`）读 `arxiv-digest` 标签下各 issue 评论的 reaction → 写 `feedback.jsonl`（每篇评论里埋 `<!-- paper:id -->` 做映射）→ commit 回仓库 → 下次排序生效。reaction 是幂等持久状态，可随时改票。
- **调度**：GitHub Actions cron，每天 08:00 America/Los_Angeles、仅工作日（见 `.github/workflows/daily.yml`）。

## 配置（GitHub Actions Secrets）
`LLM_PROVIDER`(=anthropic) · `LLM_API_KEY` · `SEED_AUTHOR` · `SMTP_SENDER_EMAIL` · `SMTP_APP_PASSWORD` · `DIGEST_RECIPIENT`
（`GITHUB_TOKEN` 由 Actions 自动注入，无需手动配；权限在 workflow 里声明 `issues: write`。）

## 手动跑（本机，会自动读同目录 `.secrets.env`）
```bash
pip install -r requirements.txt
# 只看结果不发任何东西：
python arxiv_daily.py --cats cs.CL,cs.LG,cs.AI --top 7 --dry-run
# 真发：设了 SMTP_* 就发邮件；设了 GITHUB_TOKEN + GH_REPO(owner/repo) 就建 issue
python arxiv_daily.py --cats cs.CL,cs.LG,cs.AI --top 7
```

## 改种子 / 分类 / 篇数
编辑 `.github/workflows/daily.yml` 的 `--cats` / `--top`，种子作者改 `SEED_AUTHOR` secret。
换关键词种子：`--keywords "LLM inference, quantization, ..."`；用 md 笔记：`--notes /path`。
