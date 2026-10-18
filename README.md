# NYT 美国经济新闻采集

本项目按固定版本关键词字典，采集 2010-01-01 至指定截止日的美国经济相关新闻。程序不会按情绪、政治人物或热门程度截取样本；一篇文章可以命中多个议题组，但只保留一条清洗记录。

## 方法选择

- **NYT 主通道：Archive API。** 每月一次获取完整的当月文章元数据，再对标题、摘要、snippet、导语、关键词和地理标签进行本地匹配。这避免 Article Search 每个查询最多 1,000 条造成的系统性截断。
- **NYT 补充通道：Article Search API。** 试采样默认同时运行，用 Lucene `fq` 的 OR 短语查询检查正文索引能否补到只在正文命中的文章。它仍不返回全文，因此所有 NYT 记录最多标为 `headline_abstract_lead`。
- **美国相关性。** 使用元数据、主题与地理标签、美国机构词和州/城市词联合判断，输出 `relevant / irrelevant / uncertain` 和理由。`uncertain` 不会被删除。

官方参考：

- [NYT Archive API](https://github.com/nytimes/public_api_specs/blob/master/archive_api/archive_api.md)
- [NYT Article Search API](https://github.com/nytimes/public_api_specs/blob/master/article_search/article_search_v2.yaml)

## 配置

激活环境：

```powershell
& "C:\Users\Jerry Cai\anaconda3\shell\condabin\conda-hook.ps1"
conda activate nlp-final-assignment
```

复制 `.env.example` 中的变量名到 `.env`，填入 NYT 密钥：

```env
NYT_API_KEY=...
```

验证配置（不发 API 请求）：

```powershell
python -m news_pipeline validate
```

## 运行顺序

先运行跨年份试采样。它选择 2010、2015、2020、2022、2025 和运行当年的 1 月/7 月（运行当年改为 1 月/截止月份）。当前默认来源是 NYT，并测试 NYT Archive + Article Search：

```powershell
python -m news_pipeline pilot
```

人工检查 `data/reports/` 和清洗记录后，运行全时期采集：

```powershell
python -m news_pipeline collect --start 2010-01-01
```

全量 NYT 默认只用 Archive；如已确认需要正文索引补查且配额足够，可显式使用：

```powershell
python -m news_pipeline collect --start 2010-01-01 --nyt-mode archive+search
```

NYT 官方规范公开每日额度并以 429 表示分钟或每日限额，但没有在当前规范中给出统一的每分钟数字。实测本项目密钥在快速连续请求时触发 429，因此默认使用保守的 12.5 秒间隔；可用 `.env` 中的 `NYT_MIN_INTERVAL_SECONDS` 覆盖。

如需显式选择来源：

```powershell
python -m news_pipeline collect --source nyt
```

根据 2026-10-14 的项目决定，Guardian 已停用；其原始响应、清洗文章与状态记录已删除。采集命令只接受 NYT。

如需清除某来源的既有数据，可使用：

```powershell
python -m news_pipeline purge-source guardian
```

## 输出

- `config/keywords.v1.json`：冻结版本的检索词典。
- `data/raw/nyt/YYYY-MM/*.json.gz`：带检索参数（已去除密钥）的原始 NYT API 响应。
- `data/clean/news_articles.jsonl`、`.csv`：合并去重后的主数据。
- `data/clean/analysis_candidates.jsonl`：`relevant + uncertain`，保留待人工复核项的分析候选集。
- `data/clean/relevant_only.jsonl`：仅自动规则判定为 `relevant` 的保守分析集。
- `data/clean/headlines_only.csv`：仅标题分析版本。
- `data/clean/full_text_subset.jsonl`：预留的全文分析子样本；NYT API 不提供全文，因此当前为空。
- `data/logs/collection.jsonl`：运行级日志，不含密钥。
- `data/state/manifest.jsonl`：每个区间、组和页面的完成/缓存/失败/截断状态。
- `data/reports/monthly_source_coverage.csv`：月份 × 来源的数量、正文率、相关性和不完整状态。
- `data/reports/topic_counts.csv`、`topic_overlap.csv`：议题命中和重叠；计数表同时给出全检索、relevant 与 relevant+uncertain 口径。
- `data/reports/collection_status.csv`：分页和缺失审计表。
- `data/reports/raw_response_diagnostics.csv`：每个缓存响应的 API 总命中数、页数、当前页与返回条数，用于发现上限和历史缺口。
- `data/reports/manual_review_sample.csv`：每个“来源 × 月份”确定性抽取最多 5 条，供人工填写正面/负面/中性和相关性复核；抽样本身不按情绪筛选。
- `data/reports/collection_quality.md`：质量与完整性限制。

## 重要解释限制

NYT API 不提供文章全文，标题、摘要、snippet 和导语不能视为全文。自动美国相关性规则偏保守，`uncertain` 项需要人工抽查。后续计算议题占比时，应在每个月内以一致规则获得并去重的相关经济报道数为分母；多标签占比之和可以超过 100%。
