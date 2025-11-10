import os
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import random 

from distance_metrics import euclidean_distance

def load_all_descriptors(descriptor_path='descriptors/', bins=8):
    descriptors = {}
    
    # Match pattern: '_hist_8.npy'
    desc_files = [f for f in os.listdir(descriptor_path) 
                  if f.endswith(f'_hist_{bins}.npy')]
    
    print(f"Found {len(desc_files)} descriptor files")
    if len(desc_files) > 0:
        print(f"Example filename: {desc_files[0]}")
    
    for desc_file in desc_files:
        desc_path = os.path.join(descriptor_path, desc_file)
        descriptor = np.load(desc_path)
        
        # Convert: '1_1_s_hist_8.npy' → '1_1_s.bmp'
        img_filename = desc_file.replace(f'_hist_{bins}.npy', '.bmp')
        descriptors[img_filename] = descriptor
    
    return descriptors

def visual_search(query_img, descriptors, dataset_path='dataset/Images/', 
                  distance_func=euclidean_distance, top_k=20):
    if query_img not in descriptors:
        raise ValueError(f"Query image {query_img} not found in descriptors")
    
    query_desc = descriptors[query_img]
    
    # Compute distances to all images
    distances = []
    for img_name, img_desc in descriptors.items():
        if img_name == query_img:
            continue  # Skip query image itself
        
        dist = distance_func(query_desc, img_desc)
        distances.append((img_name, dist))
    
    # Sort by distance (ascending - smaller is more similar)
    distances.sort(key=lambda x: x[1])
    
    return distances[:top_k]

def display_results(query_img, results, dataset_path='dataset/Images/', top_n=10):
    # Limit to 10 results to fit the 2×6 grid
    top_n = min(top_n, 10)
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle(f'Visual Search Results for: {query_img}', fontsize=16)

    # Display query image at (0, 0)
    query_path = os.path.join(dataset_path, query_img)
    query_image = cv2.imread(query_path)
    
    if query_image is None:
        print(f"Warning: Could not load query image: {query_path}")
        return
    
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(query_image)
    axes[0, 0].set_title('QUERY', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')

    # Hide empty space at (0, 1)
    axes[0, 1].axis('off')

    # Display top results
    for idx, (img_name, distance) in enumerate(results[:top_n]):
        # Calculate grid position
        if idx < 4:
            # First 4 results: row 0, columns 2-5
            row = 0
            col = idx + 2
        else:
            # Remaining 6 results: row 1, columns 0-5
            row = 1
            col = idx - 4
        
        # Load and display image
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not load image: {img_path}")
            axes[row, col].axis('off')
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'#{idx+1}\nDist: {distance:.3f}', 
                                 fontsize=9)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/search_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualization displayed and saved!")

def main():
    # Configuration
    DATASET_PATH = 'dataset/Images/'
    DESCRIPTOR_PATH = 'descriptors/'
    BINS = 8

    # Create results directory
    os.makedirs('results', exist_ok=True)

    print("Loading descriptors...")
    descriptors = load_all_descriptors(DESCRIPTOR_PATH, bins=BINS)
    print(f"Loaded {len(descriptors)} descriptors")

    if len(descriptors) == 0:
        print("No descriptors found!")
        print("Please run: python3 code/cvpr_computedescriptors.py first")
        return

    # Pick a random query image (or specify one)
    query_img = random.choice(list(descriptors.keys()))
    # OR specify: query_img = "1_1_s.bmp"

    print(f"\nQuery image: {query_img}")
    print("Computing distances...")

    # Perform search
    results = visual_search(
        query_img, 
        descriptors, 
        dataset_path=DATASET_PATH,
        distance_func=euclidean_distance,
        top_k=20
    )

    # Display results
    print("\nTop 10 results:")
    for idx, (img_name, distance) in enumerate(results[:10], 1):
        print(f"{idx:2d}. {img_name:20s} - Distance: {distance:.4f}")
    
    # Visualize
    display_results(query_img, results, DATASET_PATH, top_n=10)
    print("\nResults saved to: results/search_results.png")

if __name__ == "__main__":
    main()