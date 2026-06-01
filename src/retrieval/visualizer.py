import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
from PIL import Image
from src.database.models import MetadataManager


class FaissVisualizer:
    def __init__(self, metadata_manager: MetadataManager, product_images_dir: str):
        self.metadata = metadata_manager
        self.product_images_dir = product_images_dir

    def show_result(
        self,
        query_img: Image.Image,
        fname: str,
        top_matches: List[Dict[str, Any]],
    ) -> None:
        n = len(top_matches) + 1
        fig, axs = plt.subplots(1, n, figsize=(6 * n, 6))
        axs[0].imshow(query_img)
        axs[0].set_title(f"Query\n{fname}", fontsize=9)
        axs[0].axis("off")
        for k, match in enumerate(top_matches):
            pid = match["product_id"]
            score = match["score"]
            details = self.metadata.get_details(pid)
            title = details.get("title", "N/A")
            price = details.get("price_display_amount", "N/A")
            desc = details.get("description", "")[:150]
            prod_path = os.path.join(self.product_images_dir, f"{pid}.jpg")
            ax_idx = k + 1
            if os.path.exists(prod_path):
                axs[ax_idx].imshow(Image.open(prod_path).convert("RGB"))
            axs[ax_idx].set_title(f"#{k + 1}  {score:.3f}", fontsize=9)
            axs[ax_idx].axis("off")
            fig.text(
                0.2 + 0.25 * k,
                0.02,
                f"{title}\n₹{price}\n{desc}",
                ha="center",
                fontsize=8,
                wrap=True,
            )
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        print("  [Close the plot window to continue…]")
        plt.show()
