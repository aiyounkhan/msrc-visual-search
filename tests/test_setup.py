import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Testing image loading 
dataset_path = 'dataset/Images'

images = [f for f in os.listdir(dataset_path) if f.endswith('.bmp')]
print(f'Found {len(images)} images')

if len(images) > 0:
    # Load and display first image
    img_path = os.path.join(dataset_path, images[0])
    img = cv2.imread(img_path)

    print(f"Image shape: {img.shape}")  # Should be (height, width, 3)
    print(f"Image dtype: {img.dtype}")  # Should be uint8

    # Convert BRG to RBB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display
    plt.imshow(img_rgb)
    plt.title(f'Test image: {images[0]}')
    plt.axis('off')
    plt.show()

    print('Setup successful')
else: 
    print('ERROR: No images found!')
