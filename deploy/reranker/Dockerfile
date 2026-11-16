FROM python:3.12-slim
WORKDIR /service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip check
COPY server.py .
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8011"]
