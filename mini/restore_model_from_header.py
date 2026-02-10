import re
from pathlib import Path

HEADER_PATH = Path("/Users/chenee/Documents/Dev/算法/myEMG/generic-neuromotor-interface/mini/model_data.h")
OUT_PATH = Path("/Users/chenee/Documents/Dev/算法/myEMG/generic-neuromotor-interface/mini/model_data.tflite")


def main() -> None:
    text = HEADER_PATH.read_text(encoding="utf-8", errors="ignore")
    # Extract first byte array initializer
    m = re.search(r"const\s+unsigned\s+char\s+\w+\s*\[\s*\]\s*=\s*\{(.*?)\};", text, re.S)
    if not m:
        raise SystemExit("No byte array found in header.")
    body = m.group(1)
    # Find all hex bytes like 0x1c
    hex_bytes = re.findall(r"0x([0-9a-fA-F]{2})", body)
    if not hex_bytes:
        raise SystemExit("No hex bytes found.")
    data = bytes(int(b, 16) for b in hex_bytes)
    OUT_PATH.write_bytes(data)
    print(f"Wrote {len(data)} bytes to {OUT_PATH}")


if __name__ == "__main__":
    main()
