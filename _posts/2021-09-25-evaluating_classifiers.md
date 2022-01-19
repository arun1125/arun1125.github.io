# Evaluating Classifiers 



I found that the best way to go through an idea is to have an easy, low dimensional example so I can see the mechanics of whats going on. 

We're going to see that evaluating classifiers can be incredibly nuanced, and context dependant. Maybe you're building a model to see whether someone has a very rare form of cancer or just building a model to predict if someone likes football. The consequences of making mistakes and the distribution of our labels are DRASTICALLY different.

```python
#the setup
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### Context

The data we have is about a rare genetic disease. Very few people have it and our task is to build a machine learning model to figure out who has the disease and who doesn't.

nb: the model we build will be a very simple model, no feature engineering, no hyperparameter tuning, no cross validation. We are trying to see how to _evaluate_ classifiers! Maybe in the future i'll add a post with really involved modelling efforts

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
```

```python
#generate sample data and throw into dataframe
data, labels= make_classification(
    n_samples = 1000, 
    n_features = 3, 
    n_informative=3,
    n_redundant=0,
    n_classes = 2, 
    weights=[0.99]
)
data = np.append(data, labels.reshape(-1,1), axis = 1)
df = pd.DataFrame(data, columns = ['feature_0', 'feature_1', 'feature_2', 'target'])
```

```python
#split our data into train and test 
train, test = train_test_split(df,
                test_size = 0.2,
                stratify = df['target'],
                random_state = 42)

X_train, y_train = train.drop('target', axis = 1), train['target']

X_test, y_test = test.drop('target', axis = 1), test['target']

#instantiate and fit our model
model = LogisticRegression()
model.fit(X_train, y_train)
```




    LogisticRegression()



# Evaluating our model

So we've built our model, it might be rubbish and very crude right now but it exists only to server a purpose and not actually try and save someones life

```python
#get the predictions for our test set
y_preds = model.predict(X_test)
```

```python
accuracy_score(y_test, y_preds)
```




    0.985



## The problem with JUST accuracy

WOW! won't you look at that, we scored ~99% on our test set. That's all folks! go home, get on your linked in, call yourself a _dAtA sCiEnTisT_ and start applying for jobs! We crushed it! 

Alas if it were only that easy. 

A key skill you'll need in data science is skepticsim, if the results look too good, they probably are. A Healthy amount of skepticsim will get you far, especially if you start using it as a spidey sense for digging into things.

So lets have a look at our actual predictions

```python
#Get a frequency count of our predictions
Counter(y_preds)
```




    Counter({0.0: 200})



wait a second .. why are we never predicting that a patient has this super rare disease that's life threatening? something feels wrong here. 

Lets look at the definition of accuracy first

Accuracy = $\Large \frac{Number of correct Predictions}{Total number of predictions}$

A quick thought experiment: 

If our model was a child taking a test, whats the best way for this child to cheat on said test if the child was only being graded on accuracy and looking back at past papers over 99% of the answers were 0/False? 

Just Guess everything as 0/False! don't bother studying and spend that free time playing video games

But this isn't a child, its a statistical model and its predicting whether someone has a very rare life threatening illness so we can't accept this!

## different types of 'wrong' or errors! 



1. False Positive (type 1 error in the hypothesis testing context)
2. False Negative (type 2 error in the hypothesis testing context)

- (maybe ill write a separate article on hypothesis testing, A/B testing and bayesian A/B testing)

These errors sound like what they are, a False positive error, from here on will be referred to as FP is when we incorrectly classify a 0 as a 1, it's a _False_ _positive_ 

similarly a False Negative, from here on out will be reffered to as FN, is when you incorrectly classify a 1 as a 0 it's a _False_ _Negative_

### The confusion Matrix

The confusion matrix is an easy way to represent these four quantities, True Positive (TP), True Negative(TN), FP, FN

```python
cm = pd.DataFrame(confusion_matrix(y_test, y_preds), 
             columns = ['Predicted False', 'Predicted True'],
            index = ['Actual False', 'Actual True'])
cm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted False</th>
      <th>Predicted True</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual False</th>
      <td>197</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Actual True</th>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



- top left cell is TN
- top right cell is FP
- bottom left cell is FN
- bottom right cell is TP

### Different error metrics 

Our confusion matrix is a nice segway into introducing other error metrics which we can use to evaluate our classifier. Depending on the context the cost of missclassifiying a positive sample might be really high. For example in our situation we really don't want to missclassify a positive sample just in case someone actually does have this very rare disease that could kill them. 

- **Precision**: Out of all the positive values we predicted how many were actually positive
    - $\large \frac{TP}{TP+FP}$


- **Recall**: Out of all the _actual_ positive values how many did we get right? 
    - $\large \frac{TP}{TP+FN}$

```python
from sklearn.metrics import precision_score, recall_score
```

```python
precision_score(y_test, y_preds)
```

    /Users/arun/Projects/blog/venv/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))





    0.0



```python
recall_score(y_test, y_preds)
```




    0.0



Wow. With these error metrics we did AWFULLY. Honestly, we didn't even predict any true values so they were always going to be rubbish. 

How can we deal with class imbalance? (I will not go over this in detail but will list out a few points and some links for you to follow)
- Random Over/Under Sampling of minority/majority class, respectfully

- More sophistacted Over/Under Sampling techniques based on point interpolation in the feature space to generate new synthetic samples, refer the to the imblearn package for many many different ways to sample: https://imbalanced-learn.org/stable/

- Assigning different weights to the minority class - i.e telling our model, hey PAY ATTENTION TO THIS, ITS IMPORTANT!

```python
y_probs = model.predict_proba(X_test)
```

```python
from sklearn.metrics import roc_auc_score, roc_curve
```

```python
y_test
```




    480    0.0
    729    0.0
    623    0.0
    391    0.0
    100    0.0
          ... 
    48     0.0
    419    0.0
    35     0.0
    712    0.0
    980    0.0
    Name: target, Length: 200, dtype: float64



```python
roc_auc_score(y_test, y_probs[:, 1])
```




    0.83248730964467



```python
roc_curve(y_test, y_probs[:, 1])
```




    (array([0.        , 0.00507614, 0.02538071, 0.02538071, 0.04568528,
            0.04568528, 0.43147208, 0.43147208, 1.        ]),
     array([0.        , 0.        , 0.        , 0.33333333, 0.33333333,
            0.66666667, 0.66666667, 1.        , 1.        ]),
     array([1.07778106, 0.07778106, 0.04525172, 0.04492263, 0.03758554,
            0.03732233, 0.0140653 , 0.01406037, 0.00212909]))



```python
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
```

```python
plot_roc_curve(model, X_test, y_test)
```

    /Users/arun/Projects/blog/venv/lib/python3.9/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function :func:`plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: :meth:`sklearn.metric.RocCurveDisplay.from_predictions` or :meth:`sklearn.metric.RocCurveDisplay.from_estimator`.
      warnings.warn(msg, category=FutureWarning)





    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x136b988b0>




![png](evaluating_classifiers_files/output_30_2.png)


```python
plot_precision_recall_curve(model, X_test, y_test)
```

    /Users/arun/Projects/blog/venv/lib/python3.9/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_precision_recall_curve is deprecated; Function `plot_precision_recall_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: PrecisionRecallDisplay.from_predictions or PrecisionRecallDisplay.from_estimator.
      warnings.warn(msg, category=FutureWarning)





    <sklearn.metrics._plot.precision_recall_curve.PrecisionRecallDisplay at 0x136bead90>




![png](evaluating_classifiers_files/output_31_2.png)


### Moving Forward I will be slightly adjusting the context to illustrate different concepts in regards to evaluating classifiers

We are now a pub in England and want to give our local members some discounts when the football (not soccer) is on to increase our sales volume, we're going to do this buy trying to figure out which of our customers are actually football fans and sending them special coupons to redeem

```python
#generate sample data and throw into dataframe
data, labels= make_classification(
    n_samples = 1000, 
    n_features = 3, 
    n_informative=3,
    n_redundant=0,
    n_classes = 2, 
    weights=[0.65],
    random_state=42
)
data = np.append(data, labels.reshape(-1,1), axis = 1)
df = pd.DataFrame(data, columns = ['feature_0', 'feature_1', 'feature_2', 'target'])

#split our data into train and test 
train, test = train_test_split(df,
                test_size = 0.2,
                stratify = df['target'],
                random_state = 42)

X_train, y_train = train.drop('target', axis = 1), train['target']

X_test, y_test = test.drop('target', axis = 1), test['target']

#instantiate and fit our model
model = LogisticRegression()
model.fit(X_train, y_train)
```




    LogisticRegression()



```python
y_preds = model.predict(X_test)
```

```python
accuracy_score(y_test, y_preds)
```




    0.93



```python
pd.DataFrame(confusion_matrix(y_test, y_preds), 
             columns = ['Predicted False', 'Predicted True'],
            index = ['Actual False', 'Actual True'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted False</th>
      <th>Predicted True</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual False</th>
      <td>126</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Actual True</th>
      <td>10</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>



instead of calculating our confusion matrix, sklearn offers a clasification report function which calculates everything we would want to know

```python
print(classification_report(y_test, y_preds))
```

                  precision    recall  f1-score   support
    
             0.0       0.93      0.97      0.95       130
             1.0       0.94      0.86      0.90        70
    
        accuracy                           0.93       200
       macro avg       0.93      0.91      0.92       200
    weighted avg       0.93      0.93      0.93       200
    


### Probabilities

How does a classifier score the 0's and 1's? It actually assigns each sample in the test set with a probability of that sample being in each of the classes (here just 0 and 1). If the probability is greater than 0.5 then we assign it a label of 1 otherwise 0. 

Probabilities are more robust to work with as we can choose our own cut off points 

