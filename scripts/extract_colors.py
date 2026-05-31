import os
import csv
from pathlib import Path

def extract_colors():
    root_dir = Path(__file__).parent.parent
    input_csv = os.path.join(root_dir, "data", "shopify_data", "product_data.csv")
    output_csv = os.path.join(root_dir, "data", "shopify_data", "id_vs_colour.csv")
    
    if not os.path.exists(input_csv):
        print(f"Input CSV not found at {input_csv}")
        return

    print(f"Reading from {input_csv}...")
    
    with open(input_csv, mode='r', encoding='utf-8') as infile, \
         open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=['id', 'colour'])
        
        writer.writeheader()
        
        count = 0
        missing = 0
        
        for row in reader:
            pid = row['id'].strip()
            tags = row.get('product_tags', '')
            
            color_found = False
            if pid and tags:
                for tag in tags.split(','):
                    tag = tag.strip()
                    if tag.startswith('Colour:'):
                        color_name = tag.split(':')[1].strip()
                        writer.writerow({
                            'id': pid,
                            'colour': color_name
                        })
                        color_found = True
                        count += 1
                        break
                        
            if not color_found:
                missing += 1
                
        print(f"Successfully extracted {count} colors to {output_csv}")
        if missing > 0:
            print(f"Warning: {missing} products did not have a 'Colour:' tag.")

if __name__ == "__main__":
    extract_colors()
