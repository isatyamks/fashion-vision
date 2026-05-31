import logging
import os

logger = logging.getLogger(__name__)

import shutil
from io import BytesIO
from typing import Tuple

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from configs.config import PRODUCT_IMAGES_DIR, SHOPIFY_URL_CSV


class ProductImageDownloader:
    def __init__(
        self,
        url_csv: str = str(SHOPIFY_URL_CSV),
        output_dir: str = str(PRODUCT_IMAGES_DIR),
        timeout: int = 10,
        batch_size: int = 100,
    ):
        self.url_csv = url_csv
        self.output_dir = output_dir
        self.timeout = timeout
        self.batch_size = batch_size
        os.makedirs(self.output_dir, exist_ok=True)

    def download_all(self) -> Tuple[int, int, int]:
        df = pd.read_csv(self.url_csv)
        downloaded = skipped = failed = 0
        
        batch_idx = 1
        current_batch_unique_ids = set()
        current_batch_dir = self._get_batch_dir(batch_idx)
        os.makedirs(current_batch_dir, exist_ok=True)

        product_image_counts = {}

        existing_images = set()
        for root, _, files in os.walk(self.output_dir):
            for f in files:
                if f.endswith(".jpg"):
                    existing_images.add(os.path.splitext(f)[0])

        for _, row in tqdm(df.iterrows(), total=len(df)):
            prod_id = str(row['id'])
            
            if prod_id not in current_batch_unique_ids and len(current_batch_unique_ids) >= self.batch_size:
                batch_idx += 1
                current_batch_unique_ids = set()
                current_batch_dir = self._get_batch_dir(batch_idx)
                os.makedirs(current_batch_dir, exist_ok=True)
                
            current_batch_unique_ids.add(prod_id)
            product_image_counts[prod_id] = product_image_counts.get(prod_id, 0) + 1
            img_idx = product_image_counts[prod_id]
            
            out_path = os.path.join(current_batch_dir, f"{prod_id}_{img_idx}.jpg")
            
            if f"{prod_id}_{img_idx}" in existing_images:
                skipped += 1
                continue
                
            if self._download(str(row["image_url"]), out_path):
                downloaded += 1
            else:
                failed += 1

        logger.info(f"downloaded={downloaded} skipped={skipped} failed={failed}")
        return downloaded, skipped, failed

    def _get_batch_dir(self, batch_idx: int) -> str:
        return os.path.join(self.output_dir, f"batch_{batch_idx:03d}")

    def _download(self, url: str, out_path: str) -> bool:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code}: {url}")
                return False
            Image.open(BytesIO(resp.content)).convert("RGB").save(out_path, format="JPEG", quality=85)
            return True
        except Exception as e:
            logger.error(f"Error: {url}: {e}")
            return False
