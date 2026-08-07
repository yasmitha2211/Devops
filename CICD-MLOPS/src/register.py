import os
from huggingface_hub import HfApi

# Read values from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

if not HF_REPO_ID:
    raise ValueError("HF_REPO_ID environment variable is not set.")

api = HfApi(token=HF_TOKEN)

# Create repository if it doesn't exist
api.create_repo(
    repo_id=HF_REPO_ID,
    repo_type="model",
    exist_ok=True
)

# Upload the trained model
api.upload_file(
    path_or_fileobj="models/model.pkl",
    path_in_repo="model.pkl",
    repo_id=HF_REPO_ID,
    repo_type="model"
)

print("Model uploaded successfully to Hugging Face!")