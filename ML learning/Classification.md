We are going to use **MNIST** dataset : which  contains a set of 70,000 small images of digits handwritten by high school students.

Scikit Learn provides  many function to download popular datasets, where MNIST is one them.

using :
```py
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
```

`sklearn.datasets` contains mostly 3 types of functions :

1. `fetch_*` ex: `fetch_openml`- to download real-life datasets
2. `load_*` to load small toy data bundled with Scikit-Learn
3. `make_*` to generate a fake datasets,useful for tests

 The generated data is formatted like this:
`X,y = input data, target`
in NumPy arrays

Other dataset return `sklearn.utils.Bunch` objects which dictionaries which are accessed by:
`"DESCR"`
	A description of dataset
`"data"`
	The input data as a 2D array
`"target"`
	The labels, usually as 1D NumPy array

Since `fetch_openml` function by default returns the input as Pandas DataFrame and Labels the as Pandas Series.
