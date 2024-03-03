# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Update apt and install build essentials
# RUN apt-get update && \
#     apt-get install -y build-essential && \
#     rm -rf /var/lib/apt/lists/* \
#     apt install python3-pip \
#     pip3 install pywin32
ENV REPLICATE_TOKEN=r8_7q8ANOAmycausrTWroNLQJU2jqet9X62QqVna
ENV GOOGLE_API_KEY=AIzaSyCFSfTBDONxIqPd8Z9NBAtnE3CwkU6pM0A
ENV QDRANT_CLIENT=https://575f0a2b-0707-4548-85f0-bef41b244b49.us-east4-0.gcp.cloud.qdrant.io:6333
ENV QDRANT_API_KEY=P7uuBXTmnxCkzCkEtBm5pef_p0SczNz9BDfo19SHQOVPFSqu1wxPDg

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip3 install -r requirements.txt

# Expose port 8000 to the outside world
EXPOSE 8001

# Run uvicorn when the container launches
CMD ["python","main.py"]
