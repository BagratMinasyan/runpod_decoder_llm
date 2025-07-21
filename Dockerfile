FROM runpod/pytorch:0.7.0-cu1263-torch271-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      libgl1 \
      libglx-mesa0 \
      python3-pip \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . ./

CMD ["python3", "-u", "rp_handler.py"]