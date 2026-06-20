import sqlite3, json

from config import DB_PATH

c = sqlite3.connect(DB_PATH)

total_scored = c.execute(
    "SELECT COUNT(*) FROM jobs WHERE llm_score IS NOT NULL").fetchone()[0]
total_null = c.execute(
    "SELECT COUNT(*) FROM jobs WHERE llm_score IS NULL").fetchone()[0]

bad = []
for jid, title, sc, fl in c.execute(
        "SELECT id, title, llm_score, flags FROM jobs "
        "WHERE flags NOT IN ('','[]') AND llm_score IS NOT NULL"):
    try:
        caps = [f["cap"] for f in json.loads(fl)]
    except Exception:
        continue
    if caps and sc > min(caps):          # 分数高于应有封顶 = 旧逻辑漏封顶
        bad.append((title, sc, min(caps)))

print(f"已打分: {total_scored} 条")
print(f"未打分(待 backfill): {total_null} 条")
print(f"漏封顶需重评: {len(bad)} 条")
for title, sc, cap in bad[:20]:
    print(f"   • {(title or '')[:45]:45s} 现分{sc:.0f} 应≤{cap}")

c.close()