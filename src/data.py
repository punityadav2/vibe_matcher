
from typing import List, Dict
import pandas as pd

def get_mock_products() -> pd.DataFrame:
    products = [
        {"product_id":"P001","name":"Boho Maxi Dress",
         "description":"Flowy, earthy tones maxi dress perfect for festival vibes. Features paisley prints, bell sleeves, relaxed silhouette.",
         "price":89.99,"category":"Dresses","vibe_tags":["boho","festival","earthy","relaxed"]},
        {"product_id":"P002","name":"Urban Leather Jacket",
         "description":"Sleek black leather moto jacket with asymmetric zipper and silver hardware. Edgy, rebellious urban streetwear essential.",
         "price":249.99,"category":"Outerwear","vibe_tags":["urban","edgy","energetic","street"]},
        {"product_id":"P003","name":"Cozy Oversized Sweater",
         "description":"Chunky knit oversized sweater in warm cream tones. Soft, cozy, perfect for lazy weekends and coffee shop hangs.",
         "price":68.00,"category":"Tops","vibe_tags":["cozy","comfortable","relaxed","casual"]},
        {"product_id":"P004","name":"Minimalist Blazer",
         "description":"Tailored single-breasted blazer in crisp white. Clean lines, structured silhouette. Professional and sophisticated.",
         "price":159.99,"category":"Outerwear","vibe_tags":["professional","minimalist","sophisticated","formal"]},
        {"product_id":"P005","name":"Vintage Denim Jeans",
         "description":"High-waisted mom jeans with vintage wash and distressed details. Nostalgic 90s throwback with relaxed fit.",
         "price":79.99,"category":"Bottoms","vibe_tags":["vintage","casual","retro","street"]},
        {"product_id":"P006","name":"Athletic Yoga Set",
         "description":"Moisture-wicking yoga set in energetic coral. High-performance athletic wear with sculpting compression.",
         "price":95.00,"category":"Activewear","vibe_tags":["athletic","energetic","active","wellness"]},
        {"product_id":"P007","name":"Romantic Silk Blouse",
         "description":"Delicate silk blouse with ruffles and soft pink hue. Feminine, elegant, great for date nights and special occasions.",
         "price":125.00,"category":"Tops","vibe_tags":["romantic","feminine","elegant","delicate"]},
        {"product_id":"P008","name":"Grunge Plaid Shirt",
         "description":"Oversized flannel plaid shirt in dark red and black. Alternative grunge aesthetic with raw edges.",
         "price":55.00,"category":"Tops","vibe_tags":["grunge","alternative","edgy","casual"]},
        {"product_id":"P009","name":"Tropical Print Jumpsuit",
         "description":"Vibrant tropical leaf print wide-leg jumpsuit. Bold, playful vacation vibes for beach and summer adventures.",
         "price":110.00,"category":"Jumpsuits","vibe_tags":["tropical","playful","vacation","vibrant"]},
        {"product_id":"P010","name":"Sleek Monochrome Pants",
         "description":"High-waisted tailored trousers in jet black. Modern minimalist design, versatile and chic.",
         "price":89.00,"category":"Bottoms","vibe_tags":["minimalist","professional","urban","chic"]},
    ]
    df = pd.DataFrame(products)
  
    df["embedding"] = [[] for _ in range(len(df))]
    return df

if __name__ == "__main__":
    df = get_mock_products()
    print(df[["product_id","name","category","price"]].to_string(index=False))
