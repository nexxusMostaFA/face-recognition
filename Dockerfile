# Use an official Python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port 7000
EXPOSE 7000

# Run app
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7000"]
