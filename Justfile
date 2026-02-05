set shell := ["zsh", "-lc"]

lock:
    uv lock

lock-update:
    uv lock --upgrade
