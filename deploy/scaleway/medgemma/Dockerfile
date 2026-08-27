FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/scaleway/medgemma/requirements.txt /app/
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt
COPY deploy/scaleway/medgemma/app.py /app/
COPY deploy/scaleway/medgemma/contracts.py /app/
COPY src/mristudy/medgemma/output_contract.py /app/
COPY src/mristudy/schemas/knee-meniscus-finding-v2.schema.json /app/

EXPOSE 8000
CMD ["python3", "/app/app.py"]
