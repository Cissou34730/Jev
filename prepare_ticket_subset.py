"""Create a deterministic ticket subset while preserving the original schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="write a deterministic ticket subset")
    parser.add_argument("--input", default="tickets-1000.json")
    parser.add_argument("--output", default="tickets-200-minilm.json")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.offset < 0:
        raise SystemExit("--offset must not be negative")

    source = Path(args.input)
    target = Path(args.output)
    data = json.loads(source.read_text(encoding="utf-8"))
    tickets = data.get("tickets")
    if not isinstance(tickets, list):
        raise SystemExit(f"{source} has no top-level tickets array")
    end = args.offset + args.count
    if len(tickets) < end:
        raise SystemExit(
            f"{source} contains {len(tickets)} tickets, fewer than requested end index {end}"
        )

    subset = dict(data)
    subset["tickets"] = tickets[args.offset:end]
    target.write_text(
        json.dumps(subset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {target}: {len(subset['tickets'])} ticket(s)")


if __name__ == "__main__":
    main()
