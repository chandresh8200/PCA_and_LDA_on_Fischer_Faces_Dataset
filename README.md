

-----

# PCA and LDA on the Fischer Faces Dataset for Emotion Classification

This project demonstrates a two-stage dimensionality reduction technique for classifying emotions (happy vs. sad) from the Fischer Faces Dataset. The process involves using Principal Component Analysis (PCA) to reduce the initial high-dimensional image data, followed by Linear Discriminant Analysis (LDA) to find a 1-dimensional feature space that best separates the two classes.

The implementations of PCA and LDA are built from scratch using `NumPy` to showcase the underlying mathematical operations.

## Project Overview

The core objective is to classify facial images as either 'happy' or 'sad'. Each image is initially represented as a high-dimensional vector of $10201$ features (from a 101x101 pixel image).

The methodology follows these key steps:

1.  **PCA for Dimensionality Reduction:** The initial $10201$-dimensional space is reduced to an optimal intermediate dimension, $K$.
2.  **Optimizing K:** The optimal value for $K$ is determined by finding which dimension maximizes the class separability (Fisher's criterion) after applying LDA.
3.  **LDA for Class Separation:** The $K$-dimensional data is then projected onto a single dimension using LDA, maximizing the separation between the 'happy' and 'sad' classes.
4.  **Classification:** A simple threshold-based classifier is used on the final 1D data to predict the emotion and evaluate the model's accuracy.

## Methodology

### 1\. Data Preparation

  - The dataset consists of `.gif` images of faces.
  - There are 20 images for training and 10 images for testing.
  - Each image is loaded, flattened into a $10201$-dimensional vector, and assigned a label:
      - `happy`: 1
      - `sad`: 0
  - This results in a training set of shape `(20, 10201)` and a test set of shape `(10, 10201)`.

### 2\. Principal Component Analysis (PCA)

PCA is used to reduce dimensionality by projecting the data onto a lower-dimensional subspace spanned by the principal components (eigenvectors of the covariance matrix). Since the number of features ($D=10201$) is much larger than the number of samples ($N=20$), a common computational trick is employed by calculating the eigenvectors of $X X^T$ instead of the full covariance matrix $X^T X$.

### 3\. Finding the Optimal Number of Components ($K$)

A crucial step is to determine the optimal number of principal components ($K$) to retain. This was done by iterating through values of $K$ (from 2 to 19) and performing the following for each value:

1.  Reduce the training data from $10201$ to $K$ dimensions using PCA.
2.  Apply LDA to the $K$-dimensional data to project it onto 1 dimension.
3.  Calculate the class separability (Fisher's Criterion) on this 1D data:
    $$J(w) = \frac{S_B}{S_W} = \frac{(\mu_1 - \mu_0)^2}{\sigma_1^2 + \sigma_0^2}$$
    The value of $K$ that resulted in the maximum separability was chosen as the optimal one.

### 4\. Linear Discriminant Analysis (LDA)

After reducing the data to $K=18$ dimensions with PCA, LDA is applied to find the optimal projection vector ($w$) that maximizes the ratio of between-class scatter ($S_B$) to within-class scatter ($S_W$). This projects the data onto a single, highly discriminative dimension.

## Results

  - **Optimal PCA Components:** The analysis found that reducing the dimensionality to **$K=18$** components via PCA yielded the maximum class separability of **4.645** for the training data.
  - **Test Accuracy:** The model achieved an accuracy of **80%** on the unseen test set using a simple threshold classifier on the final 1D projected data.

### Visualizations

The final 1D projections for the training and test sets show a clear separation between the two classes.

**Training Data Projection (K=18)**

**Test Data Projection (K=18)**

*(Note: You can save the plots from your notebook as `training_plot.png` and `test_plot.png` and add them to your repository for these images to render.)*

## How to Run the Project

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/chandresh8200/PCA_and_LDA_on_Fischer_faces_Dataset.git
    cd PCA_and_LDA_on_Fischer_faces_Dataset
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup:**

      - Place the `Data.tar.gz` file in a location accessible by the notebook.
      - Update the file path in the notebook if necessary:
        ```python
        tar_gz_file = '/path/to/your/Data.tar.gz'
        ```

4.  **Run the Jupyter Notebook:**
    Launch Jupyter and run the cells in the `PCA_and_LDA_on_Fischer_Faces_Dataset.ipynb` notebook.

## File Structure

```
.
├── PCA_and_LDA_on_Fischer_Faces_Dataset.ipynb  # The main Jupyter Notebook
├── Data.tar.gz                               # The compressed data file (needs to be provided)
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

## Libraries Used

  - **NumPy:** For all numerical computations, including matrix operations, mean, covariance, and eigenvalue decomposition.
  - **Pillow (PIL):** For reading and processing the `.gif` image files.
  - **Matplotlib:** For plotting the 1D scatter plots to visualize class separation.
  - **tarfile & os:** For file extraction and directory management.

-----
