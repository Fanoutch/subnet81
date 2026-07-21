# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update -qq && apt-get install -y -qq \
        python3.12 python3.12-venv python3-pip \
        git build-essential wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/reliquary-venv
ENV PATH="/opt/reliquary-venv/bin:${PATH}"

# torch 2.7.0 + CUDA 12.8: matches the H100/H200 driver line shipped on
# Targon and Prime Intellect today, and is the line vLLM 0.10.x is built
# against.
RUN pip install --upgrade pip wheel setuptools \
 && pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# flash-attn prebuilt wheel for torch 2.7 / cu12 / cp312 / cxx11abi=TRUE.
# Do NOT rename the wheel on download: pip parses the version and platform
# tags from the filename.
ARG FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN wget -q "${FA_URL}" -P /tmp/ \
 && pip install /tmp/flash_attn-*.whl \
 && rm /tmp/flash_attn-*.whl

WORKDIR /opt/reliquary
COPY . /opt/reliquary
RUN pip install -e .

# vLLM 0.10.2 is the validated combination for torch 2.7 + cu128 + Qwen3.
# vLLM 0.11+ silently bumps torch to 2.8 (drops cu12 cudnn) and breaks
# the flash-attn wheel installed above.
RUN pip install 'vllm==0.10.2'

# vLLM 0.10.x calls Tokenizer.all_special_tokens_extended which transformers
# 5.x removed; pin to the 4.x line until vLLM updates its tokenizer wrapper.
RUN pip install 'transformers<5.0'

# bittensor 10.2 ships with async-substrate-interface 2.x which conflicts
# with its own scalecodec import path — roll back to the 1.x line.
RUN pip uninstall -y cyscale \
 && pip install 'async-substrate-interface<2.0.0' \
 && pip install --force-reinstall --no-deps scalecodec==1.2.12

# boto3 for R2 reads (rolled-up rollouts dataset, optional).
RUN pip install boto3

ENV GRAIL_ATTN_IMPL=flash_attention_2
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
