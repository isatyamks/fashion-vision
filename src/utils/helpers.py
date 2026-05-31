import os
from io import BytesIO
from typing import List, Tuple

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from src.utils.config import SHOPIFY_URL_SMALL_CSV


class URLValidator:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def validate_csv(self, csv_path: str = str(SHOPIFY_URL_SMALL_CSV)) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(csv_path)
        good: List[str] = []
        bad: List[str] = []

        for url in tqdm(df["image_url"].unique()):
            ok, reason = self._check(url)
            if ok:
                good.append(url)
            else:
                print(f"bad ({reason}): {url}")
                bad.append(url)

        print(f"total={len(good) + len(bad)} good={len(good)} bad={len(bad)}")
        return good, bad

    def save_report(self, bad_urls: List[str], output_path: str = "outputs/bad_urls.csv") -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame(bad_urls, columns=["bad_image_url"]).to_csv(output_path, index=False)

    def _check(self, url: str) -> Tuple[bool, str]:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=self.timeout)
            if resp.status_code != 200:
                return False, f"HTTP {resp.status_code}"
            Image.open(BytesIO(resp.content)).verify()
            return True, "ok"
        except Exception as e:
            return False, str(e)
