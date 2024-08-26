FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

# Update pip and install packages using pre-built wheels
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --prefer-binary

COPY . .

CMD ["python", "main.py"]