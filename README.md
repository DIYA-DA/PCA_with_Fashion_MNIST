# 📌 PCA with Fashion MNIST

## 🎯 Goal of This Project

In this project, we work with the Fashion MNIST dataset which contains images of clothes and accessories.  
The main goal of this project is to understand how PCA (Principal Component Analysis) works on image data.

This project includes:

- Understanding the dataset using visualization
- Applying PCA for dimensionality reduction
- Visualizing data in 2D and 3D
- Compressing images using PCA
- Reconstructing images after compression
- Checking how much information is preserved

This project helps in learning PCA step-by-step in a practical way.

---

## 🧱 Step-by-Step Explanation

### 📦 Import Libraries and Setup

We start by importing required libraries like NumPy, Pandas, and Matplotlib.  
These libraries are used for numerical computation, dataset handling, and visualization.

Matplotlib settings are used so that plots can be shown inside the notebook.

---

### 📂 Load and Filter Dataset

We load the Fashion MNIST dataset from a CSV file.  
Each row represents one image of size 28×28 pixels stored as 784 values.

Each image also has a label representing the type of clothing.

For simplicity, we keep only selected classes:

- T-shirt
- Trouser
- Sandal
- Bag
- Ankle Boot

After loading, labels are separated from image data and images are reshaped to 28×28.

---

### 🖼️ Show Example Image

A sample image from the dataset is displayed in grayscale.  
The label of the image is also shown.

This step confirms that the dataset is loaded correctly.

---

### ⚖️ Standardize the Data

Before applying PCA, the data is standardized.

Standardization means:

- Mean = 0
- Standard deviation = 1

This step is necessary because PCA is sensitive to scale.  
Without normalization, pixels with larger values would affect the result more.

---

### 🧮 Compute Covariance Matrix

The covariance matrix shows how pixels change together.

- Diagonal values represent variance
- Other values represent relationships between pixels

This matrix is required to find principal components.

---

### 📈 Eigen Decomposition

Eigenvalues and eigenvectors are computed from the covariance matrix.

- Eigenvectors represent directions of maximum variation
- Eigenvalues represent how much variation exists in those directions

Eigenvalues are plotted to see how much variance each component explains.

This helps decide how many components should be kept.

---

### 🔍 Variance Explained

We calculate how much total variance is explained by the top components.

Example:

If 50 components explain 90% variance,  
we can keep only 50 instead of all 784.

This is the main idea of PCA.

---

### 📉 Project Data onto Principal Components

The original data is projected onto the new PCA space.

Now each image is represented using fewer values.

This makes the data smaller but keeps most important information.

---

### 📊 2D PCA Visualization

Data is plotted using the first two principal components.

Each point represents one image.

This helps visualize how different classes are separated.

---

### 🌌 3D PCA Visualization

Data is also visualized using three principal components.

3D visualization shows the structure of the dataset more clearly.

Different classes can be seen as clusters.

---

### 🗜️ Compression using PCA

Instead of using all 784 pixels, only the top components are kept.

For example:

350 components instead of 784.

This reduces the size of data while preserving most information.

This process is called dimensionality reduction.

---

### 🔁 Reconstruction from Compressed Data

After compression, the images are reconstructed.

Reconstructed images are not exactly the same,
but they should look very similar to the original.

This shows how much information PCA preserved.

---

### 🖼️ Compare Original and Reconstructed Images

Original and reconstructed images are shown side by side.

If they look similar → PCA worked well  
If they look very different → too much information was lost

---

## 🚀 Conclusion

This project demonstrates how PCA can:

- Reduce dimensionality
- Preserve important information
- Help in visualization
- Compress image data
- Improve preprocessing for Machine Learning

PCA is widely used in:

- Machine Learning
- Computer Vision
- Data Science
- Deep Learning

Understanding PCA is very important for working with high-dimensional data.

---

## 👩‍💻 Author

Diya Patel  

BCA Student | Machine Learning Enthusiast  

⭐ If you like this project, consider giving it a star.
