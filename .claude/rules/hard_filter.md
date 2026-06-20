# hard_filter 设计理由(改 `hard_filter.py` 前必读)

## 它存在的原因
宽搜会灌进大量"关键词形似、实质淘汰"的岗(护理、生物台架、销售)。这些岗 JD 里
research/AI/data/PhD 等词高频,embedding 余弦和 LLM base 分都容易虚高。hard_filter
是确定性兜底,在打分前把这些拦下,不让它们污染高分段。

## 两类 flag
- **knockout(cap 5,出局)**:法定/硬性不可逾越——`clearance`(安全许可)、
  `citizenship`(排他性要求澳籍)、`registration`(受监管执业注册)、
  `deadline_passed`(截止已过)、`spam_or_test`(LLM 自检的无效岗)。
- **warn(封顶但保留可见可审计)**:领域强错配但非法定门槛——`degree_field`(20)、
  `wet_lab`(25)、`clinical_delivery`(30)。设计取向是"宁可保留可审计,别静默丢弃",
  业主可自行决定是否把某项升级为 knockout。

## 关键判定逻辑(动正则前理解,别破坏)
### citizenship —— 必须区分排他 vs 包容
"must be an Australian citizen" → 淘汰;但 "citizens, permanent residents, or …
working rights" 是**包容性**表述(PR 可投),**绝不能触发**。按句扫描:同句出现
permanent resident / working rights / residing in Australia 即判为包容,跳过。
(天真正则会在这里错杀,务必保留句级区分。)

### registration —— 标题职业 + 义务语气双路,降低误杀
历史教训:最初写成"出现 'registered nurse' 字样即 knockout",会误杀正文顺带提到
registered nurse 的好岗(如 "VR training for nurses" 研究岗)。改成两路:
1. `_REG_PROFESSION_TITLE`:岗位标题本身是受监管职业 → 该岗就是这个职业,knockout。
2. `_REG_OBLIGATION`:正文有"要求候选人持注册"的义务语气(ahpra / must be registered /
   current registration as|with 等),而非仅提及职业名。
**正例必须保持通过**:标题为 Research Fellow/Engineer、正文顺带提注册职业的好岗不触发。

### 正则边界陷阱(踩过的坑)
`speech patholog` 等词干后接 "ist" 时,词组尾部加 `\b` 会因 "patholog|ist" 间无词边界
而漏匹配。`_REG_PROFESSION_TITLE` 故意**不加尾部 `\b`**,让 patholog/physiolog 等词干
能匹配 -ist/-y 变体。改这条正则时别又把 `\b` 加回去。

## 词表从严是业主要求(不要放宽)
健康执业职业(nurse/psychologist/physio/OT/speech patholog/exercise physiolog/
clinician/registrar/radiographer/sonographer/rehabilitation consultant 等)在标题即
knockout。业主取向:"宁可多拦健康执业岗"。极小概率误伤健康×技术交叉研究岗,但那类
岗领域分本就低,代价可接受。**放宽任何健康词表前必须问业主。**

`wet_lab` 已覆盖:stem cell/cell culture/disease model/protein structure/
crystallography/bioanalytical/cryo-EM 相关/preclinical/drug discovery 等。仍有长尾
生物子领域漏网属**有意取舍**——交给 LLM 的 `unrelated_domain`/`wet_lab` 自检兜底,
不追求正则全覆盖(词表无限膨胀且脆)。

## deadline 解析的脆弱性
在 close 关键词附近 ±120 字窗口、±400 天范围内找日期,confidently 解析到的过期日
才 knockout,evidence 里打印解析出的日期供人工核。若担心错杀,可把 `deadline_passed`
降为 warn(低 cap)而非 knockout。

## 测试约定
改任何正则后,跑 `hard_filter.py` 的 `__main__` 自测,并补一个"好岗正控"
(XR/HRI/数字健康 AI/UX 各一)确认零误伤。三个历史真实误报案例(Flinders 湿实验室、
Deakin allied health、Macquarie AHPRA)应分别得 5/30/5。
