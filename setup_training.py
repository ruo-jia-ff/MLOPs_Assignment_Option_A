from datautils import get_all_images, validate_and_cleanup_images, batch_remove_background_and_make_bw, split_dataset
from glob import glob

image_folder = "Image_Dataset"

print("Gathering image paths...")
img_paths = get_all_images(image_folder)
print(f"Found {len(img_paths)} images.")

print("Validating and cleaning up bad images...")
img_paths = validate_and_cleanup_images(img_paths)

print("Removing background and converting to grayscale...")
batch_remove_background_and_make_bw(img_paths, use_gpu=True, rotate=False)

print("Splitting dataset into train/test...")
split_dataset(image_folder, test_size=0.15)