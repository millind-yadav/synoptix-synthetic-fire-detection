FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/

VOLUME ["/app/data", "/app/results"]

ENTRYPOINT ["python", "src/main.py"]
