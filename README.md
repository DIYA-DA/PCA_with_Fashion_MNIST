# PCA_with_Fashion_MNIST

🎯 Goal of This Code
You're working with images of clothes and accessories, trying to:

Understand the data with visualizations.

Reduce its dimensionality using PCA (less data, same meaning).

Visualize in 2D and 3D using principal components.

Compress and then reconstruct images to see how much information is preserved.

🧱 STEP-BY-STEP EXPLANATION
📦 1. Import Libraries & Setup
%matplotlib notebook
%matplotlib inline
These are magic commands for showing plots directly in the notebook.

You only need one of these. Use %matplotlib inline for static plots or %matplotlib notebook for interactive ones.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
These are standard Python libraries for plotting (matplotlib), numerical work (numpy), and working with tables (pandas).

📂 2. Load & Filter Dataset
data = pd.read_csv(r"C:\Users\diyap\Downloads\fashion-mnist.csv\fashion-mnist_train.csv")
Load the Fashion MNIST training dataset — a CSV file where each row is an image (flattened 28x28 = 784 values) and a label (e.g., 0 for T-shirt, 1 for Trouser).

selected_labels = [0, 1, 5, 8, 9]
data = data.loc[data['label'].isin(selected_labels)].reset_index(drop=True)
Only keep images with labels 0, 1, 5, 8, 9 for simplicity (T-shirt, Trouser, Sandal, Bag, Ankle Boot).

labels = data.pop('label')
data = data.values
images = np.reshape(data, (-1, 28, 28))
Separate the labels.

Convert image data to NumPy arrays and reshape from flat 784 to 28x28 image.

🖼️ 3. Show an Example Image
plt.imshow(images[index].squeeze(), cmap = 'gray')
plt.title(classes[labels[index]])
Show the first image in grayscale.

Label the image using the classes dictionary.

⚖️ 4. Standardize the Data
X_scaled = (data - mean_) / std_
Standardize each feature (pixel) so all are on the same scale (mean = 0, std = 1).

Why? Because PCA is sensitive to scale. Bigger numbers get more weight if not normalized.

🧮 5. Compute Covariance Matrix
cov_matrix = np.cov(features)
Measures how much each pair of pixels changes together.

Diagonal = variance of each pixel.

Off-diagonal = how two pixels change together.

📈 6. Eigen Decomposition
eig_values, eig_vectors = np.linalg.eig(cov_matrix)
Breaks the covariance matrix into:

Eigenvectors: directions of the most variation (principal components).

Eigenvalues: how much variance each direction explains.

plt.stem(eig_values[:200])
Plot the first 200 eigenvalues (called a scree plot) to see how much variance each principal component explains.

🔍 7. Check Variance Explained
for i in range(200):
    exp_var = np.sum(eig_values[:i+1])*100 / np.sum(eig_values)
Helps decide how many components to keep.

Example: if 50 components explain 90% of variance, maybe keep just 50 instead of 784.

📉 8. Project Data onto Top Components (PCA)
projected_1 = X_scaled.dot(eig_vectors.T[0])  # PC1
projected_2 = X_scaled.dot(eig_vectors.T[1])  # PC2
Project images into a new space defined by top eigenvectors (principal components).

📊 9. 2D PCA Visualization
plt.scatter(x, y, label=label_name)
Plot images as points in 2D using PC1 and PC2.

Helps visualize how well classes separate after PCA.

🌌 10. 3D PCA Visualization
res3d['PC1'], 'PC2', 'PC3', 'Y'
Like the 2D case, but now using PC1, PC2, and PC3.

Creates a 3D scatter plot of the images in reduced space.

🗜️ 11. Compression (Dimensionality Reduction)
reduced_eigen_space = eig_vectors[:, :350]
X_compressed = np.dot(X_scaled, reduced_eigen_space)
Keep only the top 350 components (instead of all 784).

This compresses each image to a lower-dimensional representation.

🔁 12. Reconstruction from Compressed Data
X_reconstructed = np.dot(X_compressed, reduced_eigen_space.T)
Recreate the full-sized data from the 350 components.

It won’t be exact, but should be visually close to the original.

🖼️ 13. Show Original vs. Reconstructed Image
plt.imshow(images[rec_index].squeeze(), cmap = 'gray')  # Original
plt.imshow(reconstructed_images[rec_index].squeeze(), cmap = 'gray')  # Reconstructed
Compares the original and compressed+reconstructed image side-by-side.

✅ Summary (In Simple Words)
Step	What It Means
Load + Show Data	Load fashion images, show examples
Scale Data	Make all pixel values equally important
Covariance + Eigenvectors	Find directions (PCs) where data varies most
Project + Visualize	See data in 2D/3D using top principal components
Compress	Keep only top 350 directions (from 784)
Reconstruct	Try to rebuild original image from reduced version
Compare	See how close reconstructed image looks to the original

