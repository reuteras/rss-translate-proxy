# LibreTranslate's image with current Debian security updates. Upstream builds on
# a pinned python:3.11.14-slim-bookworm and does not upgrade packages in its final
# stage, so v1.10.0 shipped openssl 3.0.18-1~deb12u2 while bookworm-security had
# 3.0.22. The nightly server-bin run rebuilds this image without cache.
# latest is built from main, so it can be newer than the last release
# (2026-11-24: latest = v1.10.0); pinned by digest so updates arrive as
# Dependabot PRs.
FROM docker.io/libretranslate/libretranslate:latest@sha256:e8e858a1f1db7bef3b80531d2bbef0762aa1ed7143f4565f5f5254585e69912a

USER 0
# hadolint ignore=DL3005
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# upstream's libretranslate user
USER 1032

HEALTHCHECK --interval=60s --timeout=60s --start-period=120s --retries=3 \
    CMD ["/app/venv/bin/python", "scripts/healthcheck.py"]
