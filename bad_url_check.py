import requests
import pandas as pd
from PIL import Image
from io import BytesIO

df = pd.read_csv('data\\shopify_data\\url_data_small.csv')

bad_urls = []
good_urls = []

for url in df['image_url'].unique():
    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print("Bad URL (status code != 200):", url)
            bad_urls.append(url)
            continue

        try:
            img = Image.open(BytesIO(response.content))
            img.verify() 
            print("Good URL:", url)
            good_urls.append(url)
        except Exception:
            print("Bad URL (invalid image content):", url)
            bad_urls.append(url)

    except Exception as e:
        print("Bad URL (request failed):", url)
        bad_urls.append(url)

print("\nSummary:")
print(f"Total URLs checked: {len(df['image_url'].unique())}")
print(f"Good URLs: {len(good_urls)}")
print(f"Bad URLs: {len(bad_urls)}")

pd.DataFrame(bad_urls, columns=["bad_image_url"]).to_csv("bad_urls.csv", index=False)
