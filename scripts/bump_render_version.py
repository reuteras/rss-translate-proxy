#!/usr/bin/env python3
import re
from pathlib import Path


def bump_render_version(path: Path) -> str:
    data = path.read_text(encoding="utf-8")
    m = re.search(r'^\s*render_version:\s*\"?([A-Za-z0-9._-]+)\"?\s*$', data, re.M)
    current = m.group(1) if m else "v1"
    m2 = re.match(r"^(.*?)(\d+)$", current)
    if m2:
        prefix, num = m2.group(1), int(m2.group(2))
        new = f"{prefix}{num + 1}"
    else:
        new = f"{current}-1"

    if m:
        data = re.sub(
            r'(^\s*render_version:\s*)\"?[A-Za-z0-9._-]+\"?\s*$',
            rf'\1\"{new}\"',
            data,
            flags=re.M,
        )
    else:
        data = re.sub(
            r'(^cache:\s*$)',
            rf'\1\n  render_version: \"{new}\"',
            data,
            flags=re.M,
        )
    path.write_text(data, encoding="utf-8")
    return new


def main() -> None:
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        raise SystemExit("config.yaml not found")
    new = bump_render_version(cfg_path)
    print(new)


if __name__ == "__main__":
    main()
