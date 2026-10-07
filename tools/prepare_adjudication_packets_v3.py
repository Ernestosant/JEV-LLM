"""Split real adjudication evidence into readable quota-free packets."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "data/v3/staging/reviews/adjudication_packets"
    source = directory / "awaiting_initial100.jsonl"
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    inventory = {"source_sha256": hashlib.sha256(raw).hexdigest(), "count": len(rows), "packets": []}
    for offset in range(0, len(rows), 10):
        path = directory / f"initial100_{offset // 10 + 1:03d}.json"
        payload = (json.dumps(rows[offset:offset + 10], indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace differing evidence packet: {path}")
        if not path.exists():
            path.write_bytes(payload)
        inventory["packets"].append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest(), "count": len(rows[offset:offset + 10])})
    output = directory / "initial100_packets.json"
    payload = (json.dumps(inventory, indent=2) + "\n").encode("utf-8")
    if output.exists() and output.read_bytes() != payload:
        raise ValueError("Existing adjudication packet inventory differs")
    if not output.exists():
        output.write_bytes(payload)
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
