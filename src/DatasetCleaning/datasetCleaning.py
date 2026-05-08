# import zipfile

# zip_path = r"D:\Python\CVproject\archive.zip"
# extract_path = r"D:\Python\CVproject\GTSRB_data"

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)

# print("Extraction completed!")


# =========================
# SET SEED
# =========================
import random
import numpy as np

random.seed(42)
np.random.seed(42)


#HERE  to check number of classes should be 43  if less than 43 then there is a problem with the dataset and we need to re-download it and extract it again.


import os
data_path = r"D:\Projects\Computer-Vision-project-\Data\Raw\GTSRB\Train"

classes = os.listdir(data_path)
print("Number of classes:", len(classes))


#HERE  to know the number of images in each class to check if there are any classes with very few images which might cause problems during training 
# and we can decide to either remove those classes or augment the data for those classes to balance the dataset.
import pandas as pd

data = []

for cls in classes:
    path = os.path.join(data_path, cls)
    count = len(os.listdir(path))
    data.append((cls, count))

df = pd.DataFrame(data, columns=["class", "count"])
print(df)


#HERE to visualize the distribution of images across classes to identify any imbalances in the dataset 
import matplotlib.pyplot as plt
plt.figure(figsize=(15,6))
plt.bar(df["class"], df["count"])
plt.xticks(rotation=45)
plt.title("Images per Class")
plt.show()


#HERE to visualize some sample images
import cv2

for cls in classes[:5]:
    img_path = os.path.join(data_path, cls, os.listdir(os.path.join(data_path, cls))[0])
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(f"Class {cls}")
    plt.axis("off")
    plt.show()


#HERE to remove any corrupted images
from PIL import Image

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

for root, dirs, files in os.walk(data_path):
    for file in files:
        path = os.path.join(root, file)
        if not is_valid_image(path):
            print("Removing:", path)
            os.remove(path)


#HERE to remove duplicates
import hashlib

def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

seen = set()

for root, dirs, files in os.walk(data_path):
    for file in files:
        path = os.path.join(root, file)
        h = file_hash(path)

        if h in seen:
            print("Duplicate removed:", path)
            os.remove(path)
        else:
            seen.add(h)


#HERE to check empty classes
for cls in os.listdir(data_path):
    path = os.path.join(data_path, cls)
    if len(os.listdir(path)) == 0:
        print("Empty class:", cls)


#HERE to check unreadable images
for root, dirs, files in os.walk(data_path):
    for file in files:
        path = os.path.join(root, file)
        img = cv2.imread(path)
        if img is None:
            print("Unreadable:", path)
            os.remove(path)


# =========================
# CREATE BALANCED DATASET
# =========================
import shutil

source = data_path
dest = "balanced_dataset"
TARGET = 200

#  FIX: delete old folder to avoid accumulation
if os.path.exists(dest):
    shutil.rmtree(dest)

os.makedirs(dest)

for cls in os.listdir(source):
    src_folder = os.path.join(source, cls)
    images = [f for f in os.listdir(src_folder) if f.endswith(('.png','.jpg','.jpeg'))]

    selected = random.sample(images, TARGET)

    dst_folder = os.path.join(dest, cls)
    os.makedirs(dst_folder)

    for img in selected:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(dst_folder, img))


#CHECK BALANCE
for cls in os.listdir(dest):
    print(cls, "->", len(os.listdir(os.path.join(dest, cls))))


# =========================
# CREATE SPLIT STRUCTURE
# =========================
base = "dataset"

if os.path.exists(base):
    shutil.rmtree(base)

for split in ["train", "val", "test"]:
    for cls in os.listdir(dest):
        os.makedirs(os.path.join(base, split, cls))


# =========================
# SPLIT DATA
# =========================
for cls in os.listdir(dest):
    images = os.listdir(os.path.join(dest, cls))
    random.shuffle(images)

    train_imgs = images[:140]
    val_imgs = images[140:170]
    test_imgs = images[170:]

    for split, split_imgs in zip(
        ["train","val","test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        for img in split_imgs:
            shutil.copy(
                os.path.join(dest, cls, img),
                os.path.join(base, split, cls, img)
            )


#CHECK SPLIT
for split in ["train", "val", "test"]:
    print("\n", split.upper())
    for cls in sorted(os.listdir(f"{base}/{split}")):
        count = len(os.listdir(f"{base}/{split}/{cls}"))
        print(cls, "->", count)


# =========================
# RESIZE IMAGES
# =========================
SIZE = 64

for split in ["train","val","test"]:
    for cls in os.listdir(os.path.join(base, split)):
        folder = os.path.join(base, split, cls)

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, (SIZE, SIZE))
                cv2.imwrite(img_path, img)


# =========================
# CREATE CSV 
# =========================
data = []

for split in ["train","val","test"]:
    for cls in os.listdir(os.path.join(base, split)):
        for img in os.listdir(os.path.join(base, split, cls)):
            path = f"{split}/{cls}/{img}"
            data.append([path, cls])

df = pd.DataFrame(data, columns=["image_path","label"])
df.to_csv("dataset_metadata.csv", index=False)

print("CSV saved successfully!")


# =========================
# FINAL SUMMARY
# =========================
total = 0

for split in ["train","val","test"]:
    count = 0
    for cls in os.listdir(f"{base}/{split}"):
        count += len(os.listdir(f"{base}/{split}/{cls}"))
    print(split, ":", count)
    total += count

print("Total:", total)
print("Classes:", len(os.listdir(dest)))