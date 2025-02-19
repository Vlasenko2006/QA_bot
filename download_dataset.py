import requests
import os

def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)

# Create the squad directory if it does not exist
os.makedirs('./squad', exist_ok=True)

# URLs for SQuAD v1.1 dataset
train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
dev_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

# Download and save the files
download_file(train_url, "./squad/train-v1.1.json")
download_file(dev_url, "./squad/dev-v1.1.json")
