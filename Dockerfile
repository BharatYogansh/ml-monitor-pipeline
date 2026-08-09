FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake a baseline model + reference distribution into the image so the
# API can start standalone. In the docker-compose setup this file is
# also generated fresh into the shared volume before the services start.
RUN python train_model.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
