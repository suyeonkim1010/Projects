from __future__ import annotations

import pathlib
import sys


def parse_reports(raw_text: str) -> list[list[int]]:
    reports: list[list[int]] = []

    for line in raw_text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        reports.append([int(level) for level in stripped.split()])

    return reports


def is_safe(report: list[int]) -> bool:
    if len(report) < 2:
        return True

    deltas = [right - left for left, right in zip(report, report[1:])]

    if any(abs(delta) < 1 or abs(delta) > 3 for delta in deltas):
        return False

    is_increasing = all(delta > 0 for delta in deltas)
    is_decreasing = all(delta < 0 for delta in deltas)

    return is_increasing or is_decreasing


def count_safe_reports(reports: list[list[int]]) -> int:
    return sum(1 for report in reports if is_safe(report))


def read_input() -> str:
    if len(sys.argv) > 1:
        return pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")

    if not sys.stdin.isatty():
        return sys.stdin.read()

    default_path = pathlib.Path(__file__).with_name("day2_input.txt")
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")

    raise SystemExit(
        "Provide input via stdin, a file path argument, or create day2_input.txt next to this script."
    )


def main() -> None:
    raw_text = read_input()
    reports = parse_reports(raw_text)
    print(count_safe_reports(reports))


if __name__ == "__main__":
    main()
