import os
import random
import shutil

# Base paths
base_path = "dataset"
legit_path = os.path.join(base_path, "legitimate_img")
phish_path = os.path.join(base_path, "phishing_img")

train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
test_path = os.path.join(base_path, "test")

# Get image lists
legit_images = os.listdir(legit_path)
phish_images = os.listdir(phish_path)

# Shuffle
random.shuffle(legit_images)
random.shuffle(phish_images)

# Use only 350 legit images to balance
legit_images = legit_images[:350]

# Split function
def split_data(images):
    train_split = int(0.7 * len(images))
    val_split = int(0.85 * len(images))

    train = images[:train_split]
    val = images[train_split:val_split]
    test = images[val_split:]

    return train, val, test

# Split both classes
legit_train, legit_val, legit_test = split_data(legit_images)
phish_train, phish_val, phish_test = split_data(phish_images)

# Copy function
def copy_images(image_list, src_folder, dest_folder):
    for img in image_list:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(dest_folder, img)
        )

# Copy Legitimate
copy_images(legit_train, legit_path, os.path.join(train_path, "legitimate"))
copy_images(legit_val, legit_path, os.path.join(val_path, "legitimate"))
copy_images(legit_test, legit_path, os.path.join(test_path, "legitimate"))

# Copy Phishing
copy_images(phish_train, phish_path, os.path.join(train_path, "phishing"))
copy_images(phish_val, phish_path, os.path.join(val_path, "phishing"))
copy_images(phish_test, phish_path, os.path.join(test_path, "phishing"))

print("✅ Dataset split completed successfully!")