import os
import sys
import csv
import time
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path

# Add project root to sys path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.color_analyzer import ColorAnalyzer

def main():
    root_dir = Path(__file__).parent.parent
    input_csv = os.path.join(root_dir, "data", "shopify_data", "product_data.csv")
    output_csv = os.path.join(root_dir, "data", "shopify_data", "product_colors.csv")
    
    if not os.path.exists(input_csv):
        print(f"Input CSV not found at {input_csv}")
        return

    print("Initializing Color Analyzer...")
    analyzer = ColorAnalyzer(k=3)
    
    # Check what we already processed to support resuming
    processed_ids = set()
    if os.path.exists(output_csv):
        with open(output_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_ids.add(row['id'])
        print(f"Found {len(processed_ids)} already processed colors. Resuming...")
    
    # Open for appending
    file_mode = 'a' if os.path.exists(output_csv) else 'w'
    
    with open(input_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode=file_mode, encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = ['id', 'r', 'g', 'b']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        if file_mode == 'w':
            writer.writeheader()
            
        count = 0
        skipped = 0
        errors = 0
        
        for row in reader:
            pid = row['id']
            url = row.get('image_url', '')
            
            if not url or pid in processed_ids:
                skipped += 1
                continue
                
            try:
                # Download image
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    r, g, b = analyzer.get_dominant_color(img)
                    
                    writer.writerow({
                        'id': pid,
                        'r': r,
                        'g': g,
                        'b': b
                    })
                    count += 1
                    
                    if count % 10 == 0:
                        print(f"Processed {count} images...")
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                
        print(f"\nDone! Successfully processed {count} new colors.")
        print(f"Skipped {skipped} entries, {errors} errors.")

if __name__ == "__main__":
    main()
