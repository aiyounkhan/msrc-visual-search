import numpy as np

def euclidean_distance(desc1, desc2):
    return np.sqrt(np.sum((desc1 - desc2) ** 2))


def manhattan_distance(desc1, desc2):
    return np.sum(np.abs(desc1 - desc2))


def chi_squared_distance(desc1, desc2):
    eps = 1e-10  # Small value to avoid division by zero
    return np.sum((desc1 - desc2) ** 2 / (desc1 + desc2 + eps))


def histogram_intersection(desc1, desc2):
    return 1.0 - np.sum(np.minimum(desc1, desc2))


def cosine_distance(desc1, desc2):
    dot_product = np.dot(desc1, desc2)
    norm1 = np.linalg.norm(desc1)
    norm2 = np.linalg.norm(desc2)
    return 1.0 - (dot_product / (norm1 * norm2 + 1e-10))


# Test
if __name__ == "__main__":
    # Create two random descriptors
    desc1 = np.random.rand(512)
    desc1 = desc1 / desc1.sum()
    
    desc2 = np.random.rand(512)
    desc2 = desc2 / desc2.sum()
    
    print("Testing distance metrics:")
    print(f"Euclidean: {euclidean_distance(desc1, desc2):.4f}")
    print(f"Manhattan: {manhattan_distance(desc1, desc2):.4f}")
    print(f"Chi-squared: {chi_squared_distance(desc1, desc2):.4f}")
    print(f"Hist Intersection: {histogram_intersection(desc1, desc2):.4f}")
    print(f"Cosine: {cosine_distance(desc1, desc2):.4f}")