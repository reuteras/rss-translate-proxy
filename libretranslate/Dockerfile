FROM docker.io/libretranslate/libretranslate:v1.10.0@sha256:6cf78d417741c979a12ba4f9731284b1e66066dc090c8c810174a442e0926ba2

# Distro security updates the base image lags behind: v1.10.0 ships Debian 12's
# openssl 3.0.18-1~deb12u2, and bookworm-security has fixed 27 CVEs since.
# Python's _ssl links the system libssl3, so LibreTranslate uses the upgrade.
USER root
# hadolint ignore=DL3005
RUN apt-get update && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*
USER libretranslate

# Upstream's script (as in its docker-compose.yml): GET /health, passes while
# /tmp/booting.flag exists (models still loading). The script times out at 60 s.
HEALTHCHECK --interval=60s --timeout=65s --start-period=60s --retries=3 \
    CMD ["/app/venv/bin/python", "/app/scripts/healthcheck.py"]
