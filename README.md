# FindingJob

个人求职聚合 + 本地 AI 匹配排序工具。从 Adzuna API 和 Gmail Job Alerts 抓取职位，经过三层过滤后在 Streamlit 面板里展示，同时支持投递状态追踪。所有 AI 推理均在本地运行（LM Studio），数据不离开本机。

---

## 功能概览

- **多源抓取**：Adzuna 官方 API（免费）+ Gmail Job Alert 邮件解析（Seek / Indeed / LinkedIn）
- **三层精排**：
  1. **硬性淘汰**（`pipeline/hard_filter.py`）：安全许可、AHPRA 注册要求、排他性公民身份、截止已过、湿实验室、临床交付等
  2. **语义初筛**：Nomic Embed Text 向量余弦相似度（阈值 0.60）+ 级别降权
  3. **LLM 复评**：Qwen3.5-9B 对 top-50 逐条打分（0-100）+ 输出摘要 + 自检 flags + **学科乘数**
- **学科乘数**：LLM 把每岗核心学科分为三类，Python 乘以对应系数后再送 `apply_flags` 封顶：
  - `in_domain`（CS/HCI/XR/HRI/ML-AI/数字健康）→ ×1.0
  - `adjacent`（机器人/传感/人因/通用 SWE）→ ×0.8
  - `out_of_domain`（机械/土木/化工/台架实验/临床心理）→ ×0.5
- **Streamlit 面板**：双视图——「待投递」排序列表 + 「申请追踪」状态管理；侧边栏实时显示 LLM 打分进度（每条进度 + 职位名）
- **后台补评**：`pipeline/backfill_scores.py` 在后台持续给还没 LLM 分的职位补评，逐条写库；Streamlit 侧边栏每 8 秒自动刷新显示剩余计数与进度条
- **桌面窗口**：`launcher.py` 用 pywebview 打开原生桌面窗口，无需浏览器

---

## 依赖

### Python 包

```
pip install requests numpy python-dotenv streamlit pandas pywebview
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib  # Gmail（可选）
```

### 外部服务

| 服务 | 用途 | 备注 |
|------|------|------|
| [LM Studio](https://lmstudio.ai) | 本地推理端点 | 需加载两个模型（见下） |
| Adzuna API | 职位数据 | 免费，[注册拿 app_id / app_key](https://developer.adzuna.com/) |
| Gmail API | 邮件 Job Alert（可选）| 需 `credentials.json`（见下） |

### LM Studio 需加载的模型

```powershell
lms load text-embedding-nomic-embed-text-v1.5 --gpu max
lms load qwen/qwen3.5-9b --gpu max --context-length 16384
```

---

## 快速开始

### 1. 配置环境变量

在项目根目录新建 `.env`：

```
ADZUNA_APP_ID=你的_app_id
ADZUNA_APP_KEY=你的_app_key
```

### 2. 启动（桌面窗口模式）

```
run_jobfinder.bat
```

或者直接用 Streamlit：

```
streamlit run app.py
```

### 3. 一键更新

在 Streamlit 面板左侧栏点击「🔄 一键更新」，会自动：
1. 从 Adzuna 和 Gmail 抓取新职位
2. 运行硬性淘汰 → embedding 初筛 → LLM 复评
3. 写入 `jobs.db`，后台启动 `pipeline/backfill_scores.py` 补评剩余职位

---

## 文件结构

```
FindingJob/
├── config.py                   # 单一配置源：所有常量、候选人画像、日志格式
├── app.py                      # Streamlit 面板（待投递 + 申请追踪）
├── launcher.py                 # pywebview 桌面窗口封装
│
├── pipeline/                   # 核心匹配管线
│   ├── job_matcher.py          # 主管线：抓取 → 硬筛 → embedding → LLM → 落库
│   ├── hard_filter.py          # 确定性硬性淘汰层（regex，无 LLM 依赖）
│   └── backfill_scores.py      # 后台 LLM 补评守护进程
├── infrastructure/
│   └── database.py             # SQLite 连接策略、WAL、迁移与索引
│
├── sources/                    # 数据来源采集
│   ├── adzuna_search.py        # Adzuna API 搜索 + 原生职位 ID 去重
│   └── gmail_alerts.py         # Gmail OAuth + LLM 邮件解析（可选）
│
├── scripts/                    # 运维 / 审计工具
│   ├── refresh_flags.py        # 对存量数据重跑 hard_filter，刷新 flags
│   ├── diagnose.py             # 快速审计：检查是否有分数超出封顶值的记录
│   └── inspect_flags.py        # 列出所有带 flag 的职位及命中次数统计
│
├── jobs.db                     # SQLite 数据库（运行后自动生成）
├── digest.md                   # 每次更新后输出的排序结果（便于审计）
├── run_jobfinder.bat           # Windows 启动脚本（桌面窗口模式）
├── start_app.bat               # Windows 启动脚本（浏览器模式）
├── launch_app.vbs              # 静默启动（可用于计划任务）
└── .env                        # Adzuna API 密钥（不提交到 git）
```

---

## 配置说明

所有配置集中在 `config.py`，主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ADZUNA_PAGES` | `4` | 每个搜索词抓几页（× 搜索词数 = API 调用数） |
| `SCORE_THRESHOLD` | `0.6` | embedding 相似度最低阈值；过闸太多可调高到 0.62-0.65 |
| `TOP_N_FOR_LLM` | `50` | embedding 初筛后，最多送多少条给 LLM 复评 |
| `PROFILE` | *(见 config.py)* | 候选人画像，LLM 打分的核心依据，按需修改 |
| `SEARCHES` | *(见 config.py)* | Adzuna 搜索词列表 |
| `LLM_FLAG_CAPS` | *(见 config.py)* | LLM 自检 flag 对应的分数封顶值 |
| `LMSTUDIO_BASE` | `http://localhost:1234/v1` | LM Studio API 端点 |

---

## Gmail 配置（可选）

1. 在 [Google Cloud Console](https://console.cloud.google.com/) 启用 Gmail API，创建 OAuth 桌面客户端，下载 `credentials.json` 放到项目根目录
2. 在 Gmail 创建标签 `JobAlerts`，并设置过滤器将 Seek / Indeed / LinkedIn 提醒邮件打上该标签
3. 首次运行会弹出浏览器授权，之后自动用 `token.json` 刷新

若 `credentials.json` 不存在，Gmail 模块会静默跳过，不影响 Adzuna 正常运行。

---

## 运维脚本

| 命令 | 用途 |
|------|------|
| `py -m pipeline.backfill_scores` | 手动补评所有未打分职位，完成后卸载模型 |
| `py -m pipeline.backfill_scores --watch` | 守护模式，每 60 秒扫一次 |
| `py scripts/refresh_flags.py` | 干运行：预览 hard_filter 重刷会改哪些记录 |
| `py scripts/refresh_flags.py --apply` | 实际写库：刷新所有 flags，knockout 立即出局，其余清 NULL 等 backfill 重评 |
| `py scripts/diagnose.py` | 检查是否存在分数超出封顶值的异常记录 |
| `py scripts/inspect_flags.py` | 列出所有带 flag 的职位及各 flag 命中次数 |

---

## 数据库结构

`jobs.db` 中 `jobs` 表的关键字段：

| 字段 | 说明 |
|------|------|
| `id` | SHA1(title\|company)[:16]，用于去重 |
| `sim` | Nomic embedding 余弦相似度（0-1） |
| `llm_score` | Qwen 打分（0-100）；NULL = 待 backfill |
| `llm_reason` | 一句话中文理由 |
| `flags` | JSON 数组，每项含 code / label / cap / severity / evidence；含 `discipline` 条目记录学科乘数 |
| `status` | 待投 / 已投 / 面试 / 拒 / offer / DISQUALIFIED |
| `summary` | LLM 生成的要点摘要（职责 / 要求 / 待遇） |
| `note` | 用户自填备注（面试时间、联系人等） |
| `pipeline_state` | INGESTED / EMBEDDING_REJECTED / EMBEDDING_FAILED / READY_FOR_LLM / SCORED / DISQUALIFIED |
| `score_attempts` | LLM 评分尝试轮数 |
| `last_error` | 最近一次模型或管线错误，成功后清空 |

SQLite 统一使用 WAL、30 秒 busy timeout，并为 `(source, source_id)` 和后台评分队列建立索引。
所有新采集职位都会入库；低于 embedding 阈值的记录保留为 `EMBEDDING_REJECTED`，不会显示在待投递列表。
