FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      libglib2.0-0 \
      libgl1-mesa-glx \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["python3", "-u", "rp_handler.py"]
