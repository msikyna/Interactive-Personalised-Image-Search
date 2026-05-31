FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FAISS_OPT_LEVEL=generic

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip wheel "setuptools<81" && \
    pip install --no-build-isolation "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1" && \
    grep -viE '^[[:space:]]*clip[[:space:]]*@' /app/requirements.txt > /tmp/requirements-no-clip.txt && \
    pip install -r /tmp/requirements-no-clip.txt && \
    pip install gunicorn

COPY . /app

RUN mkdir -p /app/runtime /app/user_matrices /app/cache /app/media && \
    chmod +x /app/docker/entrypoint.sh

EXPOSE 8942

ENTRYPOINT ["/app/docker/entrypoint.sh"]
