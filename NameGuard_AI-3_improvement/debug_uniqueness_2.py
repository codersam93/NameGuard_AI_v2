import pandas as pd
from backend.scoring import simple_phonetic_key, similarity, DATA

name = "Super Duper Unique Name"
key = simple_phonetic_key(name)
print(f"Name: {name}, Key: {key}")

sims = [(other, similarity(key, other)) for other in DATA.existing_company_keys]
sims.sort(key=lambda x: x[1], reverse=True)

print("Top 5 similar keys:")
for k, s in sims[:5]:
    print(f"{k}: {s}")
