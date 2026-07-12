FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir \
    huggingface_hub \
    accelerate \
    sentencepiece \
    protobuf \
    einops \
    autoawq

ENV HF_HOME=/model-cache
ENV TRANSFORMERS_CACHE=/model-cache
ENV HF_HUB_CACHE=/model-cache

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

CMD ["--model", "Qwen/Qwen2.5-1.5B-Instruct", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--dtype", "bfloat16", \
     "--gpu-memory-utilization", "0.90", \
     "--max-model-len", "8192"]
