from pathlib import Path

log_dir = Path("/content/generic-neuromotor-interface/logs/2026-02-17/03-29-52")

print("日志目录结构:")
print("=" * 80)

for item in sorted(log_dir.rglob("*")):
    if item.is_file():
        size = item.stat().st_size
        rel_path = item.relative_to(log_dir)
        print(f"{rel_path} ({size:,} bytes)")
