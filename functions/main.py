import os
import requests
from google.cloud import storage, language_v1
from fastapi import HTTPException

# Environment variables set at deploy time
YOLO_URL    = os.environ["YOLO_URL"]
RECIPE_KEY  = os.environ["RECIPE_API_KEY"]

# A small set of detected classes to ignore (non‑food items)
NON_FOOD = {"bottle", "refrigerator", "couch", "bench", "chair", "dining table", 'bowl', 'cup'}

def process_image(event, context):
    """ Cloud Function entrypoint: triggered on new image in Cloud Storage. """
    name = event["name"]
    # skip our own results files
    if name.startswith("results/"):
        return

    bucket_name = event["bucket"]
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(name)

    # Download image bytes
    img_bytes = blob.download_as_bytes()

    # Call YOLO server
    resp = requests.post(f"{YOLO_URL}/predict", files={"file": img_bytes})
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from YOLO server: {resp.text}")

    preds = data.get("predictions", [])
    print("Raw preds:", preds)  # debug

    # Filter to edible ingredients only
    ingredients = sorted({
        p["class_name"]
        for p in preds
        if p.get("confidence", 0) > 0.3 and p.get("class_name") not in NON_FOOD
    })
    print("Filtered ingredients:", ingredients)

    # If no edible ingredients, bail out
    if not ingredients:
        out = "No edible ingredients detected."
        client.bucket(bucket_name).blob(f"results/{name}.txt").upload_from_string(out)
        return

    # 1) Find recipes by ingredients
    find_params = {
        "apiKey": RECIPE_KEY,
        "ingredients": ",".join(ingredients),
        "number": 5,
        "ignorePantry": True
    }
    finds = requests.get(
        "https://api.spoonacular.com/recipes/findByIngredients",
        params=find_params
    ).json()
    print("FindByIngredients results:", finds)

    if not isinstance(finds, list) or not finds:
        out = f"No recipes found for ingredients: {ingredients}"
        client.bucket(bucket_name).blob(f"results/{name}.txt").upload_from_string(out)
        return

    # 2) For each recipe, get detailed info & run sentiment on summary
    nlp = language_v1.LanguageServiceClient()
    scored = []  # list of tuples (score, title)
    for rec in finds:
        rid   = rec.get("id")
        title = rec.get("title", "Untitled")
        # fetch full info (including summary)
        info_resp = requests.get(
            f"https://api.spoonacular.com/recipes/{rid}/information",
            params={"apiKey": RECIPE_KEY}
        )
        if not info_resp.ok:
            print(f"Failed to fetch information for recipe {rid}: {info_resp.status_code}")
            continue
        info = info_resp.json()
        summary = info.get("summary", "")
        # sentiment analysis
        doc = language_v1.Document(content=summary, type_=language_v1.Document.Type.PLAIN_TEXT)
        score = nlp.analyze_sentiment(document=doc).document_sentiment.score
        scored.append((score, title))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Format output lines
    lines = [f"{title}: sentiment score {score:.2f}" for score, title in scored]

    # Upload results
    client.bucket(bucket_name).blob(f"results/{name}.txt")\
          .upload_from_string("\n".join(lines))
