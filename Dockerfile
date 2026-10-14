# syntax=docker/dockerfile:1

# ---- deps: resolve the locked runtime set into /opt/venv (wheels only, nothing is compiled) --
FROM python:3.12-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---- runtime: the venv on a fresh slim image; same base, so /opt/venv/bin/python resolves ----
FROM python:3.12-slim
RUN groupadd --system --gid 10001 search \
    && useradd --system --uid 10001 --gid 10001 --home-dir /home/search --create-home \
        --shell /usr/sbin/nologin search \
    && install -d -o 10001 -g 10001 /data
COPY --from=deps /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PRODSEARCH_DATA_DIR=/data \
    HF_HOME=/data/hf \
    NUMBA_CACHE_DIR=/tmp/numba
USER 10001:10001
WORKDIR /home/search
EXPOSE 8000
# The default command serves the bundled 40-product sample, so the image answers without any
# download. For WANDS: mount a volume on /data, run `prodsearch fetch` and `prodsearch index`
# in it, then `prodsearch serve --host 0.0.0.0`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
ENTRYPOINT ["prodsearch"]
CMD ["serve", "--sample", "--host", "0.0.0.0", "--port", "8000"]
