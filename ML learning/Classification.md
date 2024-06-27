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
That we set `as_frame = flase`.

![[Pasted image 20240627113959.png]]

There are 70,000 images and 784 Features. Because one pixel show intensity from 0(white) to 255(black)

Let us peek a one image we need grab the an instance's(image's) feature vector and rehape it to 28 x 28 array.
*feature vector: an one dimensional array (vector) have multiple elements(feature)*

We are going to use  Matplotlib's imshow():
![[Pasted image 20240627115449.png]]
What the labels tells us:
`>>> y[0]`
`'5'`

Before Inspecting that we are going to split the MNIST datasets in training set and test set.
The data set returned by fetch_openml() is already split into training set:test set::6:1
We don't need to shuffle.

## Training a Binary Classifier 
*Binary Classifier: Classifier capable to distinguish between 2 classes*

```
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
```

We are going to a using a **stochastic gradient descent(SGD)**
	*stochastic gradient descent(SGD): a* 
It can handheld very large datasets efficiently.

```
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train,y_train_5)
```
`>>> sgd_clf.predict([some_digit])`
`array([ True])`

## Performance Measures

### Measuring Accuracy Using Cross-Validation 
Using `cross_val_score()` function to evaluate the SDClassifier Model using k-fold cross-validation with 3 folds.

`>>> from sklearn.model_selection import cross_val_score`
`>>> cross_val_score(sgd_clf,X_train,y_train_5,cv = 3,scoring ="accuracy")`
`array([0.95035, 0.96035, 0.9604 ])`

Lets look at the dummy classifiers for non 5 image

```
rom sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train,y_train_5)
print(any(dummy_clf.predict(X_train)))
```
`False`

`>>> cross_val_score(dummy_clf,X_train,y_train_5,cv = 3,scoring= "accuracy")`
`array([0.90965, 0.90965, 0.90965])`

This because only 10% of the dataset is 5 and guessing that image is not 5 is right about 90%.

