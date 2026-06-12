FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        espeak-ng \
        ffmpeg \
        git \
        libsndfile1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY experiments/voice-encoder/requirements.txt /tmp/voice-encoder-requirements.txt

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && /root/.local/bin/uv python install 3.12 \
    && /root/.local/bin/uv venv --seed --python 3.12 /opt/venv \
    && /opt/venv/bin/python -m pip install -r /tmp/voice-encoder-requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"

CMD ["bash"]
