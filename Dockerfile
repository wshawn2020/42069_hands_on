# Use a lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script(s) into the container
COPY scripts/random_forest_detector.py ./scripts/

# Define the command to run your script automatically when the container starts
CMD ["python", "./scripts/random_forest_detector.py"]