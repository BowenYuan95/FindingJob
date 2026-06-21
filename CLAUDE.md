# FindingJob — 个人求职聚合与匹配工具

业主:Bowen Yuan(CS PhD,HCI/XR,2026 年中毕业,澳洲 PR)。本工具自动聚合澳洲岗位、
本地打分排序,辅助决定投哪些岗。**详细的设计理由见 `.claude/rules/`,改对应模块前先读。**

## 架构(五段管线,职责分层)
1. **采集**(`job_matcher.fetch_adzuna` + `gmail_alerts`)—— Adzuna API 宽搜召回。
2. **硬筛**(`hard_filter.scan_disqualifiers`)—— 确定性淘汰/封顶,纯正则、无 LLM。
3. **语义初筛**(embedding 余弦)—— 本地 LM Studio Nomic,过 `SCORE_THRESHOLD`。
4. **LLM 复评**(`job_matcher.llm_review`)—— 本地 Qwen,给 base 分 + 自检 flags + 摘要。
5. **异步补评**(`backfill_scores.py`)—— 空闲时把未评岗补完,封顶逻辑与主程序一致。

数据库连接、迁移、WAL 与索引统一由 `infrastructure/database.py` 管理。采集结果必须全部
入库并通过 `pipeline_state` 表达处理结果；不要重新引入“embedding 未过即静默丢弃”的逻辑。
业务编排不得新增裸 SQL 或直接 `requests` 调用；职位存取走 `JobRepository`，模型调用走
`infrastructure/lmstudio.py`。

存量维护:`refresh_flags.py` 对数据库重跑 hard_filter 刷新 flags,**不重抓 Adzuna**。

## 核心设计哲学:宽搜窄评
搜索层只管"召回够多",所有"严"的判断下沉到打分层(硬筛→embedding→LLM 三道关)。
漏召回不可恢复、噪音可恢复,故召回宁滥勿缺、精排宁严勿松。
→ 不要在搜索层加 `what_phrase`/`title_only` 这类卡召回的写法。

## 🚫 红线(改动前必须遵守,违反会破坏系统正确性)
- **封顶执行必须在 Python(`apply_flags`),绝不交回 LLM 自己算 `min(base, cap)`。**
  LLM 算术纪律不可靠,历史上正是它漏封顶导致一批岗虚高。LLM 只"检测+列 flag code"。
- **`hard_filter` 的健康/执业词表是"故意从严"的,不要擅自放宽。** 业主明确要求
  护理/allied health/执业注册类岗硬淘汰。放宽前必须问业主。
- **不要往 CV/求职信/任何材料里写未经业主确认的技能**(Docker/FastAPI/云/向量库/
  生产 RAG 等只到概念层)。"诚实桥接"是铁律:可迁移能力可自信表述,直接宣称需真凭据。
- **PR 身份硬过滤**:任何要求 Australian citizenship / NV1/NV2 / security clearance
  的岗一律淘汰(`clearance`/`citizenship` flag)。
- **修改 `llm_review` 返回签名要全链路同步**:它返回四元组
  `(score, reason, summary, llm_flags)`,`backfill_scores.py` 依赖此签名。

## flags 与封顶(单一事实来源:`hard_filter` + `job_matcher.LLM_FLAG_CAPS`)
- knockout(cap 5,直接出局):`clearance` `citizenship` `registration` `deadline_passed` `spam_or_test`
- warn(封顶但保留可见):`degree_field`(20) `wet_lab`(25) `clinical_delivery`(30)
  `seniority`(35) `unrelated_domain`(30)
- 最终分 = `min(base, min(命中 flag 的 cap))`;任一 knockout → status=DISQUALIFIED。
- 合并正则 flags + LLM 自检 flags 后必须 `dedup_flags` 去重再 `apply_flags`。

## 打分 rubric(权重 40/25/20/10/5,学术/产业五五开)
领域40 · 技能25 · 资历20 · 地点10 · 加成5。学术研究岗与产业 R&D/AI 工程岗
领域分相同不偏袒;资历分对 Level A/B 与产业 graduate/junior 同等友好。
**权重本身不随"偏学术/偏产业"调整**——五五开就是不 tilt。

## 关键参数(调参前先看真实分布,别拍脑袋)
- `SCORE_THRESHOLD`(默认 0.6):宽搜下偏松,过阈值可能数百条。收紧到 0.66 前先看 sim 分布。
- `TOP_N_FOR_LLM`(默认 50):仅首轮即时覆盖量;其余靠 backfill 补,不是总量上限。
- `ADZUNA_PAGES`(4):调用数 = 搜索条数 × 页数,留意 Adzuna 免费配额。Adzuna 后台不一定
  显示剩余次数,靠脚本内计数更可靠。

## 已知边界(不是 bug,是有意取舍)
- hard_filter 只拦高频明确的淘汰项;冷门生物/化学 bench 岗的长尾靠 LLM 语义兜底,
  不强求正则全覆盖(避免词表无限膨胀且变脆)。
- embedding 轻微偏学术(PROFILE 研究味浓),产业岗余弦略低,靠 backfill 兜底可接受。

## 环境
- Windows + PowerShell(注意:`py -c "..."` 嵌套引号常出错,复杂查询写成 `.py` 文件跑)。
- 本地 LM Studio 提供 embedding + chat 端点;Adzuna key 在 `.env`。
- 破坏性操作(refresh/删数据)默认 dry-run,业主确认后再 `--apply`。
