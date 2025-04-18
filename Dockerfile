# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the fine-tuned model directory into the container
COPY ./fine_tuned_model /app/fine_tuned_model

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the web server
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]