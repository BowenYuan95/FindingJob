import sqlite3, json
from collections import Counter

from config import DB_PATH

c = sqlite3.connect(DB_PATH)
rows = c.execute(
    "SELECT title, company, status, flags FROM jobs "
    "WHERE flags != '[]' AND flags != ''"
).fetchall()

counter = Counter()
print(f"=== 带 flag 的岗共 {len(rows)} 条 ===\n")
for title, company, status, fl in rows:
    flags = json.loads(fl)
    codes = [f["code"] for f in flags]
    counter.update(codes)
    mark = "⛔" if status == "DISQUALIFIED" else "⚠"
    print(f"{mark} {(title or '')[:45]:45s} | {(company or '')[:22]:22s} | {codes}")

print("\n=== 各 flag 命中次数 ===")
for code, n in counter.most_common():
    print(f"  {code:18s} {n}")