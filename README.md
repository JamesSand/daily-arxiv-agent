# daily-arxiv-agent

每天抓最新 arXiv 论文 → 按相关度排序 → Claude (Opus 4.8) 翻译中文摘要（中英对照）→ **Telegram 推送（每篇带 👍/👎，反馈调整后续推荐）**。

- **种子**：聚焦高效 LLM 推理方向的论文（低比特量化 / KV-cache / 稀疏注意力 / 投机解码）。具体种子作者放在 Actions Secret `SEED_AUTHOR`，不写进仓库。
- **排序**：纯 Python TF-IDF 余弦 + Rocchio 相关性反馈（`q = 种子 + 👍均值 − 👎均值`），已投票的论文不再出现。零额外依赖。
- **摘要**：Anthropic **Opus 4.8**（官方 SDK）。可用 `LLM_MODEL` 覆盖。
- **反馈闭环（零服务器）**：Telegram inline 👍/👎 按钮带 `callback_data`；每天的 Action 用 `getUpdates` 把前一天的点击拉下来 → 写 `feedback.jsonl`（`tg_offset.txt` 记游标）→ commit 回仓库 → 下次排序生效。
- **调度**：GitHub Actions cron，每天 08:00 America/Los_Angeles（见 `.github/workflows/daily.yml`）。

## 配置（GitHub Actions Secrets）
`LLM_PROVIDER`(=anthropic) · `LLM_API_KEY` · `SEED_AUTHOR` · `TELEGRAM_BOT_TOKEN` · `TELEGRAM_CHAT_ID`

## 手动跑（本机，会自动读同目录 `.secrets.env`）
```bash
pip install -r requirements.txt
python arxiv_daily.py --cats cs.CL,cs.LG,cs.AI --top 12
# 设了 TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID 就发 Telegram；否则发邮件（SMTP_*）或打印
```

## 改种子 / 分类 / 篇数
编辑 `.github/workflows/daily.yml` 的 `--cats` / `--top`，种子作者改 `SEED_AUTHOR` secret。
换关键词种子：`--keywords "LLM inference, quantization, ..."`；用 md 笔记：`--notes /path`。
