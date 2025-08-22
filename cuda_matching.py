import os
import requests
import csv
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import faiss
from concurrent.futures import ThreadPoolExecutor
import torch

# For visualization
import matplotlib.pyplot as plt


# ------------------- CONFIG -------------------
import argparse
PRODUCT_IMAGES_DIR = "product_images"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"results\\matched_results_{timestamp}.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
FAILED_URLS_LOG = "failed_urls.txt"
THRESHOLD = 0.75
BATCH_SIZE = 32  # batch size for encoding images
# ----------------------------------------------
# Parse args for crops directory
parser = argparse.ArgumentParser(description="Match cropped images to products.")
parser.add_argument('--crops_dir', type=str, required=True, help='Directory containing cropped images')
args = parser.parse_args()
CROPPED_IMAGES_DIR = args.crops_dir

# Load model on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("clip-ViT-B-32", device=device)
model.to(device)
print(f"model running on {device}")
# ------------------- HELPER -------------------

# For offline: load product image from local directory
def load_product_image_local(img_id):
    img_path = os.path.join(PRODUCT_IMAGES_DIR, f"{img_id}.jpg")
    try:
        img = Image.open(img_path).convert("RGB")
        return img, None
    except Exception as e:
        return None, img_path


# ------------------- LOAD PRODUCT DETAILS -------------------
PRODUCT_DATA_CSV = "data/shopify_data/product_data.csv"
product_details = {}
with open(PRODUCT_DATA_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        product_details[row['id']] = row

# ------------------- ENCODE PRODUCTS -------------------


print("Loading and encoding product images from local directory...")
product_embeddings = []
product_ids = []
bad_imgs = []
product_files = [f for f in os.listdir(PRODUCT_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
for fname in tqdm(product_files, total=len(product_files)):
    img_id = os.path.splitext(fname)[0]
    img_path = os.path.join(PRODUCT_IMAGES_DIR, fname)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        bad_imgs.append(img_path)
        continue
    emb = model.encode(img, normalize_embeddings=True)
    product_embeddings.append(emb)
    product_ids.append(img_id)

if bad_imgs:
    with open(FAILED_URLS_LOG, "w") as f:
        f.writelines(str(path) + "\n" for path in bad_imgs)
    print(f"Logged {len(bad_imgs)} missing product images to '{FAILED_URLS_LOG}'")

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

    # batch_embeddings = model.encode(batch_images, normalize_embeddings=True).astype("float32")
    batch_embeddings = model.encode(
    batch_images,
    normalize_embeddings=True,
    device=device).astype("float32")



    # FAISS search for top 3 matches
    top_k = 3
    distances, indices = index.search(batch_embeddings, top_k)
    for j, fname in enumerate(valid_files):
        top_matches = []
        for k in range(top_k):
            idx = indices[j][k]
            score = distances[j][k]
            if score >= THRESHOLD:
                top_matches.append((product_ids[idx], score))
        if top_matches:
            # Save only the best match for CSV output
            results.append((fname, top_matches[0][0], top_matches[0][1]))
            print(f"{fname} top matches: " + ", ".join([f"{pid} ({s:.3f})" for pid, s in top_matches]))

            # Visualization: show cropped and top 3 matched product images with details
            fig, axs = plt.subplots(1, top_k+1, figsize=(6*(top_k+1), 6))
            axs[0].imshow(batch_images[j])
            axs[0].set_title('Cropped')
            axs[0].axis('off')
            for k, (matched_id, score) in enumerate(top_matches):
                product_img, _ = load_product_image_local(matched_id)
                details = product_details.get(str(matched_id), {})
                title = details.get('title', 'N/A')
                price = details.get('price_display_amount', 'N/A')
                desc = details.get('description', 'N/A')
                if product_img is not None:
                    axs[k+1].imshow(product_img)
                    axs[k+1].set_title(f'#{k+1} Prob: {score:.3f}')
                    axs[k+1].axis('off')
                    details_text = f"{title}\nPrice: {price}\n\n{desc[:200]}{'...' if len(desc)>200 else ''}"
                    fig.text(0.25 + 0.25*k, 0.05, details_text, ha='center', va='bottom', fontsize=10, wrap=True)
                else:
                    axs[k+1].set_title(f'#{k+1} (Image not found)')
                    axs[k+1].axis('off')
            plt.tight_layout(rect=[0,0.1,1,1])
            print("Close this plot window to see the next result...")
            plt.show()
        else:
            print(f"{fname} - No confident match (top 3 scores: {[distances[j][k] for k in range(top_k)]})")

# ------------------- SAVE RESULTS -------------------
import csv
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "matched_id", "similarity"])
    writer.writerows(results)
print(f"\nDone! Matches saved to '{OUTPUT_CSV}'")
