# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .
RUN python build_db.py

# Expose the port the app runs on
EXPOSE 5001

# Command to run the app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "main:app"]
