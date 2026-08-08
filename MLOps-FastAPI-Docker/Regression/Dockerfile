# Use a small, official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements.txt first (Docker layer caching: if requirements
# don't change, Docker reuses this layer instead of reinstalling every time)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (app.py, model.pkl, schema.json, etc.)
COPY . .

# Document that the container listens on port 8000
EXPOSE 8000

# Start the Flask app. Runs as "python app.py" so app.run(host="0.0.0.0",
# port=8000) inside app.py takes effect.
CMD ["python", "app.py"]
