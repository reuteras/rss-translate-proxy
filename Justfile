set shell := ["zsh", "-lc"]

lock:
    uv lock

lock-update:
    uv lock --upgrade

bump-render-version:
    python3 scripts/bump_render_version.py
