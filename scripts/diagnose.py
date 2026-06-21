import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3, json

from config import DB_PATH
from infrastructure.database import initialize_database

c = initialize_database(DB_PATH)

total_scored = c.execute(
    "SELECT COUNT(*) FROM jobs WHERE llm_score IS NOT NULL").fetchone()[0]
total_null = c.execute(
    "SELECT COUNT(*) FROM jobs WHERE llm_score IS NULL").fetchone()[0]

bad = []
bad_status = []
for jid, title, sc, fl in c.execute(
        "SELECT id, title, llm_score, flags FROM jobs "
        "WHERE flags NOT IN ('','[]') AND llm_score IS NOT NULL"):
    try:
        flags = json.loads(fl)
        caps = [f["cap"] for f in flags]
    except Exception:
        continue
    if caps and sc > min(caps):          # 分数高于应有封顶 = 旧逻辑漏封顶
        bad.append((title, sc, min(caps)))
    if any(f.get("severity") == "knockout" for f in flags):
        status = c.execute("SELECT status FROM jobs WHERE id=?", (jid,)).fetchone()[0]
        if status != "DISQUALIFIED":
            bad_status.append((title, status))

print(f"已打分: {total_scored} 条")
print(f"未打分(待 backfill): {total_null} 条")
print(f"漏封顶需重评: {len(bad)} 条")
for title, sc, cap in bad[:20]:
    print(f"   • {(title or '')[:45]:45s} 现分{sc:.0f} 应≤{cap}")
print(f"knockout 状态不一致: {len(bad_status)} 条")
for title, status in bad_status[:20]:
    print(f"   • {(title or '')[:45]:45s} 当前状态={status}")

c.close()
