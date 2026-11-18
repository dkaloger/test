FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir fastapi uvicorn
COPY model.py serve.py ./
COPY static ./static
COPY checkpoints/model.pt ./checkpoints/model.pt
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
