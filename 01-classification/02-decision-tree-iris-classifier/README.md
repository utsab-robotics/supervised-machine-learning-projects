# Dataset Information

## Iris Flower Dataset

This project uses the Iris dataset distributed with scikit-learn.

- **Samples:** 150
- **Input features:** 4
- **Target classes:** Setosa, Versicolor, Virginica
- **Feature units:** Centimeters

### Features

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

### Loading the dataset

The training script loads the dataset directly:

```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
```

No external download is required.

### Original reference

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.

The dataset is commonly known as Fisher's Iris dataset. Check the scikit-learn documentation for dataset details and the original reference.