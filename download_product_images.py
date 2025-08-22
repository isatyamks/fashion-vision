import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

SHOPIFY_DATA = "data/shopify_data/url_data.csv"
OUTPUT_DIR = "product_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_image(url, out_path):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(out_path)
            return True
        else:
            print(f"Failed to download {url}: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    df = pd.read_csv(SHOPIFY_DATA)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_url = row['image_url']
        img_id = row['id']
        out_path = os.path.join(OUTPUT_DIR, f"{img_id}.jpg")
        if not os.path.exists(out_path):
            download_image(img_url, out_path)
        else:
            print(f"Image for ID {img_id} already exists, skipping.")

if __name__ == "__main__":
    main()
