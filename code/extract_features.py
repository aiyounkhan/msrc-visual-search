import cv2
import numpy as np 
import os

def extract_color_histogram(img_path, bins = 8):
    img = cv2.imread(img_path)

    if img is None: 
        raise ValueError(f'No image is found in {img_path}')
    
    # Convert BGR to RGB 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Quantize each channel 
    quantized = (img_rgb // (256 // bins)).astype(np.int32)

    # Make 3D histogram
    histogram = np.zeros((bins, bins, bins), dtype=np.float32)

    # Fill histogram by counting pixels
    height, width, _ = img.shape
    for i in range(height):
        for j in range(width):
            r = quantized[i, j, 0]
            g = quantized[i, j, 1]
            b = quantized[i, j, 2]
            histogram[r, g, b] += 1

    # Flatten the 1D vector 
    descriptor = histogram.flatten()

    # Normalize (sum to 1.0)
    descriptor = descriptor / descriptor.sum()

    return descriptor

# Test the function 
if __name__ == '__main__':
    # Test on one image 
    test_img = 'dataset/Images/1_1_s.bmp'

    print("Testing histogram extraction...")
    desc = extract_color_histogram(test_img, bins=8)

    print(f"Descriptor shape: {desc.shape}")
    print(f"Descriptor sum: {desc.sum():.6f}")
    print(f"First 10 values: {desc[:10]}")
    print(f"Min value: {desc.min():.6f}")
    print(f"Max value: {desc.max():.6f}")
