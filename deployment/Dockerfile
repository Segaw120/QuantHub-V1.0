FROM python:3.9-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application
COPY ./app /code/app
COPY ./models /code/models
COPY ./deployment/main_api.py /code/main_api.py

# Expose port
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "7860"]
