"""
generate_dataset.py
Generates a small synthetic image dataset (32x32 RGB, 3 classes)
saved as train/val/test splits in numpy .npz format.
"""
import numpy as np
import os

def generate_synthetic_images(n_samples=1500, img_size=32, n_classes=3, seed=42):
    np.random.seed(seed)
    X, y = [], []
    per_class = n_samples // n_classes

    for cls in range(n_classes):
        for _ in range(per_class):
            img = np.random.randn(img_size, img_size, 3).astype(np.float32)
            # Each class has a distinct color bias + texture pattern
            img[:, :, cls] += 1.5
            if cls == 0:
                # Horizontal stripes
                img[::4, :, :] += 2.0
            elif cls == 1:
                # Vertical stripes
                img[:, ::4, :] += 2.0
            else:
                # Diagonal pattern
                for i in range(img_size):
                    img[i, i % img_size, :] += 2.0
            img = np.clip(img, -3, 3)
            X.append(img)
            y.append(cls)

    X = np.array(X)
    y = np.array(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Normalize to [0, 1]
    X = (X - X.min()) / (X.max() - X.min())

    # Splits: 70/15/15
    n = len(X)
    t1 = int(0.7 * n)
    t2 = int(0.85 * n)

    splits = {
        "X_train": X[:t1], "y_train": y[:t1],
        "X_val":   X[t1:t2], "y_val":   y[t1:t2],
        "X_test":  X[t2:], "y_test":   y[t2:],
    }

    os.makedirs("data", exist_ok=True)
    np.savez("data/dataset.npz", **splits)

    print(f"Dataset saved to data/dataset.npz")
    print(f"  Train: {splits['X_train'].shape}, Val: {splits['X_val'].shape}, Test: {splits['X_test'].shape}")
    print(f"  Classes: {n_classes} | Image size: {img_size}x{img_size} RGB")
    return splits

if __name__ == "__main__":
    generate_synthetic_images()
