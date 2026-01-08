import json
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------

COCO_ANNOTATIONS = "D:/GITHUB/OD/ml-engineer-computer-vision-pipeline/data/raw/coco/annotations/instances_train2017.json"
COCO_IMAGES_DIR = "D:/GITHUB/OD/ml-engineer-computer-vision-pipeline/data/raw/coco/images/train2017"

OUTPUT_IMAGES_DIR = "D:/GITHUB/OD/ml-engineer-computer-vision-pipeline/data/processed/images"
OUTPUT_LABELS_DIR = "D:/GITHUB/OD/ml-engineer-computer-vision-pipeline/data/processed/labels"

DATASET_NAME = "coco_person_vehicle_animal"
DATASET_VERSION = "v1"

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "ml_pipeline"
MONGO_COLLECTION = "dataset_metadata"

# COCO category mappings
VEHICLE_IDS = [2, 3, 4, 6, 7, 8]
ANIMAL_IDS = list(range(16, 26))
CLASS_NAMES = ["person", "vehicle", "animal"]


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def map_category(cat_id):
    if cat_id == 1:
        return 0  # person
    if cat_id in VEHICLE_IDS:
        return 1  # vehicle
    if cat_id in ANIMAL_IDS:
        return 2  # animal
    return None


def coco_to_yolo(box, img_w, img_h):
    x, y, w, h = box
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    return xc, yc, w / img_w, h / img_h


def save_split(ids, split, records):
    for img_id in tqdm(ids, desc=f"Saving {split} data"):
        rec = records[img_id]

        src_img = os.path.join(COCO_IMAGES_DIR, rec["file"])
        out_img = os.path.join(OUTPUT_IMAGES_DIR, split, rec["file"])
        out_lbl = os.path.join(
            OUTPUT_LABELS_DIR,
            split,
            rec["file"].replace(".jpg", ".txt")
        )

        os.makedirs(os.path.dirname(out_img), exist_ok=True)
        os.makedirs(os.path.dirname(out_lbl), exist_ok=True)

        img = cv2.imread(src_img)
        if img is None:
            continue

        cv2.imwrite(out_img, img)

        with open(out_lbl, "w") as f:
            for label in rec["labels"]:
                f.write(" ".join(map(str, label)) + "\n")


# -----------------------------
# MAIN EXECUTION
# -----------------------------

def main():
    print("Loading COCO annotations...")

    with open(COCO_ANNOTATIONS, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    records = {}

    print("Processing annotations...")

    for ann in annotations:
        cls = map_category(ann["category_id"])
        if cls is None:
            continue

        img_info = images[ann["image_id"]]
        img_id = img_info["id"]

        if img_id not in records:
            records[img_id] = {
                "file": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
                "labels": []
            }

        yolo_box = coco_to_yolo(
            ann["bbox"],
            img_info["width"],
            img_info["height"]
        )

        records[img_id]["labels"].append((cls, *yolo_box))

    print(f"Total usable images: {len(records)}")

    # -----------------------------
    # TRAIN / VAL / TEST SPLIT
    # -----------------------------

    img_ids = list(records.keys())

    train_ids, temp_ids = train_test_split(
        img_ids, test_size=0.3, random_state=42
    )

    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.33, random_state=42
    )

    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # -----------------------------
    # SAVE DATASETS
    # -----------------------------

    save_split(train_ids, "train", records)
    save_split(val_ids, "val", records)
    save_split(test_ids, "test", records)

    # -----------------------------
    # MONGODB METADATA INSERTION
    # -----------------------------

    print("Inserting metadata into MongoDB...")

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    col = db[MONGO_COLLECTION]

    docs = []

    for img_id in records:
        split = (
            "train" if img_id in train_ids
            else "val" if img_id in val_ids
            else "test"
        )

        docs.append({
            "image_file": records[img_id]["file"],
            "split": split,
            "classes": CLASS_NAMES,
            "dataset": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "source": "COCO-2017",
            "created_at": datetime.utcnow()
        })

    col.insert_many(docs)

    print("Dataset preparation complete ✔")


if __name__ == "__main__":
    main()
