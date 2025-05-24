FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyQt5, MPI, and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    libopenmpi-dev \
    openmpi-bin \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Set up a virtual display for GUI components if needed
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1

# Define entry point script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "src/main.py"]
