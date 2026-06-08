# daily-arxiv-agent

每天抓最新 arXiv 论文 → 按相关度排序 → Claude (Opus 4.8) 翻译中文摘要（中英对照）→ 邮件推送。

- 种子：聚焦高效 LLM 推理方向的论文（低比特量化 / KV-cache / 稀疏注意力 / 投机解码）。具体种子作者放在 Actions Secret `SEED_AUTHOR`，不写进仓库。
- 排序：纯 Python TF-IDF 余弦（`arxiv_daily.py`，零额外依赖）。
- 摘要：Anthropic **Opus 4.8**（官方 SDK，最强模型）。可用 `LLM_MODEL` 覆盖。
- 调度：GitHub Actions cron，每天 08:00 America/Los_Angeles（见 `.github/workflows/daily.yml`）。
- 投递：Gmail SMTP。

## 配置（GitHub Actions Secrets）
`LLM_PROVIDER` (=anthropic) · `LLM_API_KEY` · `SMTP_SENDER_EMAIL` · `SMTP_APP_PASSWORD` · `DIGEST_RECIPIENT`

## 手动跑
```bash
pip install -r requirements.txt
# 本机会自动读同目录 .secrets.env；或直接用环境变量
python arxiv_daily.py --seed-author "First Last" --cats cs.CL,cs.LG,cs.AI --top 12
```

## 改种子 / 分类 / 篇数
编辑 `.github/workflows/daily.yml` 里的 `--seed-author` / `--cats` / `--top`。
换关键词种子：`--keywords "LLM inference, quantization, ..."`；用 md 笔记：`--notes /path/to/notes`。
