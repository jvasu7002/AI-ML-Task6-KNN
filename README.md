# K-Nearest Neighbors (KNN) Classification

## Project Overview

This project implements the K-Nearest Neighbors (KNN) algorithm for classification using a CSV dataset (Iris dataset).
The model predicts the class of data points based on their nearest neighbors.

---

## Objective

* Understand and implement KNN algorithm
* Normalize features for better accuracy
* Experiment with different values of K
* Evaluate model performance

---

## Tools and Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Dataset

* Dataset used: Iris Dataset (CSV format)
* Features:

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width
* Target:

  * Species (Setosa, Versicolor, Virginica)

---

## Steps Performed

1. Loaded dataset using Pandas
2. Split data into features (X) and target (y)
3. Converted categorical labels into numeric values
4. Applied train-test split
5. Normalized data using StandardScaler
6. Implemented KNN algorithm
7. Tested multiple values of K (1, 3, 5, 7)
8. Evaluated model using:

   * Accuracy Score
   * Confusion Matrix
9. Visualized decision boundary

---

## Results

* Accuracy: 100% (for all tested K values)
* Confusion Matrix shows no misclassification
* Model performs very well due to clean and well-separated dataset

---

## Visualization

* Decision Boundary graph shows how KNN separates different classes
* Each region represents a predicted class

---

## Key Learnings

* KNN is a simple and effective algorithm
* Feature scaling is very important
* Choice of K affects performance
* Works best on small and clean datasets

---

## Advantages and Disadvantages

### Advantages

* Simple to understand
* No training phase
* Works well for small datasets

### Disadvantages

* Slow for large datasets
* Sensitive to noise
* Requires proper scaling

---

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
```

```bash
python task6.py
```

---

## Project Structure

```
project/
│── task6.py
│── iris.csv
│── README.md
```

---

## Author

Vasu Jain

---

## Conclusion

KNN achieved perfect accuracy on this dataset, demonstrating its effectiveness for classification problems with well-separated data.
