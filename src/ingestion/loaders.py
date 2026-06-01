import csv
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from urllib.parse import urlparse
class ProductImageDownloader:
    def __init__(self, csv_path: str, product_dir: str):
        self.csv_path = csv_path
        self.product_dir = product_dir
        os.makedirs(self.product_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
        }
    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme in ('http', 'https'), result.netloc])
        except ValueError:
            return False
    def _parse_csv(self) -> Dict[str, List[str]]:
        products = {}
        with open(self.csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['id'].strip()
                if not pid:
                    continue
                url = row.get('image_url', '').strip()
                if url and self._is_valid_url(url):
                    if pid not in products:
                        products[pid] = []
                    products[pid].append(url)
        return products
    def _download_single(self, pid: str, idx: int, url: str, batch_dir: str) -> None:
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1]
        if not ext:
            ext = ".jpg"
        filename = f"{pid}_{idx}{ext}"
        out_path = os.path.join(batch_dir, filename)
        if os.path.exists(out_path):
            return
        print(f"Downloading {filename} to {os.path.basename(batch_dir)}...")
        try:
            resp = requests.get(url, headers=self.headers, stream=True, timeout=15)
            resp.raise_for_status()
            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception:
            print(f"Failed to download {url}")
    def download_all(self, workers: int = 15, target_batch_size: int = 100):
        products = self._parse_csv()
        batches_tasks = {}
        current_count = 0
        batch_idx = 1
        for pid, urls in products.items():
            urls = products[pid]
            if current_count >= target_batch_size:
                batch_idx += 1
                current_count = 0
            batch_name = f"batch_{batch_idx:03d}"
            batch_dir = os.path.join(self.product_dir, batch_name)
            os.makedirs(batch_dir, exist_ok=True)
            if batch_dir not in batches_tasks:
                batches_tasks[batch_dir] = []
            for idx, url in enumerate(urls, start=1):
                batches_tasks[batch_dir].append((pid, idx, url, batch_dir))
                current_count += 1
        for batch_dir, tasks in batches_tasks.items():
            bname = os.path.basename(batch_dir)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self._download_single, t[0], t[1], t[2], t[3]) for t in tasks]
                for future in as_completed(futures):
                    pass
        print("Download and batching complete.")
