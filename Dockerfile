FROM python:3.8-slim

WORKDIR /app

# Install system dependencies for RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# First uninstall any existing Flask and Werkzeug
RUN pip uninstall -y flask werkzeug

# Install specific versions known to work together
RUN pip install --no-cache-dir flask==1.1.2 werkzeug==1.0.1 jinja2==2.11.3 itsdangerous==1.1.0 click==7.1.2

# Copy requirements file and install other dependencies without upgrading Flask/Werkzeug
COPY requirements.txt .
RUN grep -v "flask\|werkzeug" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt

# Copy application code
COPY models/ ./models/
COPY templates/ ./templates/
# Create static directory if it doesn't exist
RUN mkdir -p ./static
COPY *.py ./

# Expose the port
EXPOSE 3000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application directly with Python instead of gunicorn
CMD ["python", "app.py"]