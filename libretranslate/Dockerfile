# LibreTranslate's image with current Debian security updates. Upstream builds on
# a pinned python:3.11.14-slim-bookworm and does not upgrade packages in its final
# stage, so v1.10.0 shipped openssl 3.0.18-1~deb12u2 while bookworm-security had
# 3.0.22. The nightly server-bin run rebuilds this image without cache.
# Pinned to a release tag, so the version is visible, plus its digest; new
# releases arrive as Dependabot PRs.
FROM docker.io/libretranslate/libretranslate:v1.10.0@sha256:6cf78d417741c979a12ba4f9731284b1e66066dc090c8c810174a442e0926ba2

USER 0
# hadolint ignore=DL3005
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# upstream's libretranslate user
USER 1032

HEALTHCHECK --interval=60s --timeout=60s --start-period=120s --retries=3 \
    CMD ["/app/venv/bin/python", "scripts/healthcheck.py"]
