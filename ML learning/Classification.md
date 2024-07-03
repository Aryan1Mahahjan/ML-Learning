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


We dealing with *skewed datasets* so i will be easier to user CM

#### Implementing Cross-Validation

To use a cross-validation process with more control than the cross_val_score() function.
We use:

```
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits = 3,)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct /len(y_pred))
```

```
Output:

0.95035
0.96035
0.9604
```


## Confusion Matrices

To count the number of times class A are classified as class B for All A/B pairs.

To compute the Confusion Matrix we are going to us `cross_val_predict`, because it returns the prediction made on each fold. This means the model can make clean prediction on data that it near seen before.

```
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv = 3)
```

We are now going to apply selection Matrix :

```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5,y_train_pred)
# y_train_5 - Target & y_train_pred- predicted class
```

```
>>>cm
		|
array([[53892,   687],
       [ 1891,  3530]])
```

![[Pasted image 20240628121246.png]]

Analyzing the output :

53,892 Correctly Classified as non-5s (True Negative)
687 Wrongly Classified as 5s (False Positives)
1891 Wrongly Classified as non-5s (False Negative)
3530 Correctly Classified as 5s (True Positive)

In a perfect Classifier would only have True Negative and True Positive.

## Precision & Recall

The accuracy of positive prediction is called the *precision* of a classifier.
Equation 

precision = TP/{TP+FP}

The ratio of positive instances that are correctly detected by classifier is called  *recall*.

recall = TP/{TP+FN}

Scikit Learn provides several function to compute Classifier Metric ,Include Precision and recall: 

```
>>> from sklearn.metrics import precision_score, recall_score
>>> precision_score(y_train_5,y_train_pred)
0.8370879772350012
>>> recall_score(y_train_5,y_train_pred)
0.6511713705958311
```
It claims when an image represent 5 its correct only 83.7% of the time. It only detects  65.7% of the 5s.


Often Precision and Recall are combined in  single metric called the *harmonic mean* represented by F<sub>1</sub> . The harmonic mean gives more weight to the low values.

So to get a high F<sub>1</sub> both values have to be high.
F<sub>1</sub> =   2 * { precision * recall }/{ precision + recall } 
We can compute  F<sub>1</sub> score, simply using `f1_score()`

```
>>> from sklearn.metrics import f1_score
>>> f1_score(y_train_5,y_train_pred)
0.7325171197343847
```

F<sub>1</sub> favors classifier with similar precision and recall
Using  F<sub>1</sub>  isn't always desirable, because in some case you care more about precision and and other case you care about recall.

## The Precision/Recall Trade-off

SGDClassifier makes a score based on *Decision Function*. 

If the score is greater than the threshold it assigns it positive, if it is less than the threshold then is assigns it to be negative.