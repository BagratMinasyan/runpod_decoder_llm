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

# Debug pip version and print exact installation output
RUN python3 --version && \
    pip --version && \
    echo "📦 Installing requirements..." && \
    pip install --no-cache-dir --verbose -r requirements.txt || (echo "❌ Pip install failed!" && cat requirements.txt && exit 1)

COPY . .

# Optionally confirm what's installed
RUN pip list > installed_packages.txt && cat installed_packages.txt

CMD ["python3", "-u", "rp_handler.py"]
