import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PRODUCT_IMAGES_DIR, SHOPIFY_URL_CSV
from src.ingestion.loaders import ProductImageDownloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    downloader = ProductImageDownloader(
        csv_path=str(SHOPIFY_URL_CSV),
        product_dir=str(PRODUCT_IMAGES_DIR)
    )
    downloader.download_all(workers=args.workers)

if __name__ == "__main__":
    main()
