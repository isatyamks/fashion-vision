import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime


CROPPED_IMAGES_DIR = "video_crops\\crops_2025-07-06_14-28-08"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"results\\matched_results_{timestamp}.csv"
FAILED_URLS_LOG = "failed_urls.txt"
threshold = 0.85


shoping_cart_data = "data\\shopify_data\\url_data_small.csv"       
df = pd.read_csv(shoping_cart_data)


model = SentenceTransformer("clip-ViT-B-32")

product_embeddings = []
product_ids = []
bad_urls = []

print("Encoding product images...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    url = row['image_url']
    prod_id = row['id']

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            raise ValueError(f"Status code: {response.status_code}")

        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as pil_err:
            raise ValueError(f"PIL failed to open image: {pil_err}")

        emb = model.encode(img, normalize_embeddings=True)
        product_embeddings.append(emb)
        product_ids.append(prod_id)

    except Exception as e:
        print(f"rror with {url}: {e}")
        bad_urls.append(url)

if bad_urls:
    with open(FAILED_URLS_LOG, "w") as f:
        f.writelines(url + "\n" for url in bad_urls)
    print(f"Logged {len(bad_urls)} bad URLs to '{FAILED_URLS_LOG}'")

product_embeddings = np.array(product_embeddings)

print("Matching cropped images to products...")
results = []

for fname in tqdm(os.listdir(CROPPED_IMAGES_DIR)):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    try:
        img_path = os.path.join(CROPPED_IMAGES_DIR, fname)
        img = Image.open(img_path).convert("RGB")
        emb = model.encode(img, normalize_embeddings=True).reshape(1, -1)

        sims = cosine_similarity(emb, product_embeddings)[0]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score >= threshold:
            matched_id = product_ids[best_idx]
            print(f"{fname} ID {matched_id} | Score: {best_score:.3f}")
            results.append((fname, matched_id, best_score))
        else:
            print(f"{fname} - No confident match (score: {best_score:.3f})")

    except Exception as e:
        print(f"Error with image {fname}: {e}")

pd.DataFrame(results, columns=["filename", "matched_id", "similarity"]).to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Matches saved to '{OUTPUT_CSV}'")
