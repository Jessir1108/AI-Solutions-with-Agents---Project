import os
import zipfile

import gdown

# Constants
ZIP_URL = "https://drive.google.com/uc?id=1Ife6sDb8Ay5zpKw0bMJZnUBTPczPGte7"
ZIP_PATH = "./tmp_dataset.zip"
EXTRACT_DIR = "./"

# Ensure dataset folder exists
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Download ZIP if not already present
if not os.path.exists(ZIP_PATH):
    print("📦 Downloading dataset...")
    gdown.download(ZIP_URL, ZIP_PATH, quiet=False)
else:
    print("✅ ZIP already downloaded")

# Extract contents
print("🗃️ Extracting dataset...")
with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("✅ Dataset ready in:", EXTRACT_DIR)
