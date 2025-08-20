FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV SHELL=/bin/bash

WORKDIR /app

# build-essential g++ cmake pkg-config are for pysdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ cmake pkg-config \
    python3 python3-pip python3-venv python3-dev \
    wget git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install ninja

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888