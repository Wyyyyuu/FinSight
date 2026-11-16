FROM python:3.12-slim
WORKDIR /service
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.8.0 --index-url ${TORCH_INDEX_URL}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip check
COPY server.py .
ENV HF_HOME=/models HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
EXPOSE 8010
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8010"]
