#  Cats vs Dogs Image Classification using SVM

##  Project Overview

This project implements a **Support Vector Machine (SVM)** model to classify images of **cats and dogs** using the **Kaggle Dogs vs Cats dataset**.
Since SVM works on numerical data, **HOG (Histogram of Oriented Gradients)** is used to extract meaningful features from images before classification.

---

##  Objective

* To classify images into **Cat** or **Dog**
* To understand image preprocessing and feature extraction
* To apply **SVM** for binary image classification
* To gain hands-on experience with a real-world dataset

---

##  Dataset

* **Source:** Kaggle – Dogs vs Cats Dataset
* **Images:** Labeled images of cats and dogs
* **Classes:**

  * Cat → `0`
  * Dog → `1`

### Dataset Structure (Kaggle)

```
/kaggle/input/<dataset-name>/
└── dataset/
    ├── train/
    │   ├── cat.0.jpg
    │   ├── dog.0.jpg
    │   └── ...
    └── test/
```

> Note: In this dataset, the `train` folder is inside an extra `dataset/` directory.
> The code automatically detects this structure.

---

##  Methodology

### 1. Image Loading

* Images are read from the training folder
* Only filenames starting with `cat` or `dog` are considered

### 2. Preprocessing

* Images resized to **64 × 64**
* Converted to **grayscale**

### 3. Feature Extraction

* **HOG (Histogram of Oriented Gradients)** used to extract features
* Converts images into numerical feature vectors

### 4. Model Training

* **Support Vector Machine (SVM)** with a **linear kernel**
* Dataset split into:

  * 80% training
  * 20% testing

### 5. Evaluation

* Accuracy calculated using `accuracy_score`

---

##  Technologies Used

* Python
* OpenCV
* NumPy
* scikit-image
* scikit-learn
* Kaggle Notebook

---

##  Libraries Required

```bash
numpy
opencv-python
scikit-image
scikit-learn
```

(All are pre-installed in Kaggle)

---

##  How to Run the Project (Kaggle)

1. Open a new **Kaggle Notebook**
2. Add the **Dogs vs Cats dataset** using *Add Input*
3. Copy and run the provided notebook/code
4. The model will automatically:

   * Detect dataset path
   * Load images
   * Train SVM
   * Display accuracy

---

##  Sample Output

```
Main dataset folder: /kaggle/input/your-dataset-name
Inside folder: ['dataset']
Final train path: /kaggle/input/your-dataset-name/dataset/train
Images loaded: 1000
SVM Accuracy: 80–85 %
```

---

##  Results

* Achieved **~80–85% accuracy** using SVM with HOG features
* Model performs well on limited image samples
* Demonstrates effectiveness of classical ML for image classification

---

##  Limitations

* SVM is slower with very large datasets
* Accuracy depends on feature extraction
* CNN models can achieve higher accuracy

---

##  Future Improvements

* Use **CNN (Convolutional Neural Network)** for higher accuracy
* Increase dataset size
* Try different SVM kernels (RBF)
* Add confusion matrix and visualization

---

## Learning Outcomes

* Understood image preprocessing techniques
* Learned feature extraction using HOG
* Implemented SVM for binary classification
* Worked with real-world image datasets

---

##  Conclusion

This project successfully demonstrates how **Support Vector Machines** can be applied to **image classification tasks** using **feature extraction techniques** like HOG.
It serves as a strong foundation for transitioning to deep learning models such as CNNs.

---

##  References

* Kaggle Dogs vs Cats Dataset
* scikit-learn Documentation
* OpenCV Documentation

---
