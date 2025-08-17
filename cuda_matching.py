import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import faiss
from concurrent.futures import ThreadPoolExecutor

# ------------------- CONFIG -------------------
CROPPED_IMAGES_DIR = "video_crops\\crops_2025-08-17_12-50-31"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"results\\matched_results_{timestamp}.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
FAILED_URLS_LOG = "failed_urls.txt"
SHOPIFY_DATA = "data\\shopify_data\\url_data_small.csv"
THRESHOLD = 0.85
BATCH_SIZE = 32  # batch size for encoding images
# ----------------------------------------------

# Load product data
df = pd.read_csv(SHOPIFY_DATA)

# Load model on GPU if available
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
print(f"model running on {device}")
model = SentenceTransformer("clip-ViT-B-32", device=device)

# ------------------- HELPER -------------------
def fetch_image(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"Status code: {response.status_code}")
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img, None
    except Exception as e:
        return None, url

# ------------------- ENCODE PRODUCTS -------------------
print("Fetching and encoding product images...")

product_embeddings = []
product_ids = []
bad_urls = []

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(fetch_image, row['image_url']): row for _, row in df.iterrows()}
    for future in tqdm(futures, total=len(futures)):
        row = futures[future]
        img, failed_url = future.result()
        if failed_url:
            bad_urls.append(failed_url)
            continue
        emb = model.encode(img, normalize_embeddings=True)
        product_embeddings.append(emb)
        product_ids.append(row['id'])

if bad_urls:
    with open(FAILED_URLS_LOG, "w") as f:
        f.writelines(url + "\n" for url in bad_urls)
    print(f"Logged {len(bad_urls)} bad URLs to '{FAILED_URLS_LOG}'")

product_embeddings = np.array(product_embeddings).astype("float32")

# ------------------- BUILD FAISS INDEX -------------------
print("Building FAISS index...")
dim = product_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity for normalized embeddings
index.add(product_embeddings)

# ------------------- MATCH CROPPED IMAGES -------------------
print("Matching cropped images...")

cropped_files = [f for f in os.listdir(CROPPED_IMAGES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
results = []

# Batch encode cropped images
for i in tqdm(range(0, len(cropped_files), BATCH_SIZE)):
    batch_files = cropped_files[i:i+BATCH_SIZE]
    batch_images = []
    valid_files = []

    for fname in batch_files:
        try:
            img_path = os.path.join(CROPPED_IMAGES_DIR, fname)
            img = Image.open(img_path).convert("RGB")
            batch_images.append(img)
            valid_files.append(fname)
        except:
            print(f"Failed to open cropped image: {fname}")

    if not batch_images:
        continue

    batch_embeddings = model.encode(batch_images, normalize_embeddings=True).astype("float32")

    # FAISS search
    distances, indices = index.search(batch_embeddings, 1)  # top 1 match
    for j, fname in enumerate(valid_files):
        best_idx = indices[j][0]
        score = distances[j][0]
        if score >= THRESHOLD:
            results.append((fname, product_ids[best_idx], score))
            print(f"{fname} matched ID {product_ids[best_idx]} | Score: {score:.3f}")
        else:
            print(f"{fname} - No confident match (score: {score:.3f})")

# ------------------- SAVE RESULTS -------------------
pd.DataFrame(results, columns=["filename", "matched_id", "similarity"]).to_csv(OUTPUT_CSV, index=False)
print(f"\nDone! Matches saved to '{OUTPUT_CSV}'")
