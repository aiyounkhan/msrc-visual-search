import os 
import numpy as np 
from tqdm import tqdm # This is for progess bar

from extract_features import extract_color_histogram

def compute_all_descriptors (dataset_path = 'dataset/Images/', descriptor_path = 'descriptors/', bins = 8):
    # Create the descriptor directory if it doesnot exists
    os.makedirs(descriptor_path, exist_ok=True)

    # Get all image files
    image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.bmp')])

    print(f"Found {len(image_files)} images")
    print(f"Computing descriptors with {bins} bins per channel...")
    print(f"Descriptor size: {bins**3} dimensions")

    # Process each of the images in the dataset
    for img_file in tqdm(image_files, desc = 'Processing images'):
        img_path = os.path.join(dataset_path, img_file)

        try:
            # Extract the discriptor for each image 
            descriptor = extract_color_histogram(img_path, bins)

            # Save the descriptor 
            desc_filename = img_file.replace('.bmp', f'_hist_{bins}.npy')
            desc_path = os.path.join(descriptor_path, desc_filename)
            np.save(desc_path,  descriptor)
        
        except Exception as e:
            print(f"\nError processing {img_file}: {e}")
    
    print(f"\nDone! Descriptors saved to: {descriptor_path}")
    print(f"Total descriptors: {len(os.listdir(descriptor_path))}")


if __name__ == "__main__":
    # Run with default settings
    compute_all_descriptors(
        dataset_path='dataset/Images/',
        descriptor_path='descriptors/',
        bins=8
    )   

