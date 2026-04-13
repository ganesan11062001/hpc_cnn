"""
generate_dataset.py
Downloads CIFAR-10 (or falls back to synthetic if no internet).
Saves as data/dataset.npz in the same format the training script expects.
"""
import numpy as np
import os

def generate_dataset(data_dir="data", use_cifar=True):
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "dataset.npz")

    if use_cifar:
        try:
            import torchvision
            import torch
            print(">>> Downloading CIFAR-10...")
            train = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True)
            test  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

            X_train = train.data.astype(np.float32) / 255.0   # (50000, 32, 32, 3)
            y_train = np.array(train.targets)

            X_test  = test.data.astype(np.float32) / 255.0    # (10000, 32, 32, 3)
            y_test  = np.array(test.targets)

            # Use 5000 train + 1000 val + 1000 test for faster HPO
            np.random.seed(42)
            idx = np.random.permutation(len(X_train))[:6000]
            X_train, y_train = X_train[idx], y_train[idx]

            X_val,  y_val  = X_train[5000:], y_train[5000:]
            X_train, y_train = X_train[:5000], y_train[:5000]

            tidx = np.random.permutation(len(X_test))[:1000]
            X_test, y_test = X_test[tidx], y_test[tidx]

            np.savez(out_path,
                     X_train=X_train, y_train=y_train,
                     X_val=X_val,     y_val=y_val,
                     X_test=X_test,   y_test=y_test)

            print(f"CIFAR-10 saved to {out_path}")
            print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
            print(f"  Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")

        except Exception as e:
            print(f"CIFAR-10 download failed ({e}) — falling back to synthetic dataset")
            generate_synthetic(out_path)
    else:
        generate_synthetic(out_path)

def generate_synthetic(out_path, n_samples=1500, img_size=32, n_classes=3):
    print(">>> Generating synthetic dataset...")
    np.random.seed(42)
    X, y = [], []
    for cls in range(n_classes):
        for _ in range(n_samples // n_classes):
            img = np.random.randn(img_size, img_size, 3).astype(np.float32)
            img[:, :, cls] += 1.5
            X.append(img); y.append(cls)
    X, y = np.array(X), np.array(y)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    X = (X - X.min()) / (X.max() - X.min())
    t1, t2 = int(0.7*len(X)), int(0.85*len(X))
    np.savez(out_path,
             X_train=X[:t1],  y_train=y[:t1],
             X_val=X[t1:t2],  y_val=y[t1:t2],
             X_test=X[t2:],   y_test=y[t2:])
    print(f"Synthetic dataset saved to {out_path}")

if __name__ == "__main__":
    generate_dataset(use_cifar=True)
