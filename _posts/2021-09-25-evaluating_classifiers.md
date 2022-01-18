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
train, test = train_test_split(df,
                test_size = 0.2,
                stratify = df['target'],
                random_state = 42)
```

```python
X_train, y_train = train.drop('target', axis = 1), train['target']

X_test, y_test = test.drop('target', axis = 1), test['target']
```

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```




    LogisticRegression()



# Evaluating our model

So we've built our model, it might be rubbish and very crude right now but hopefully it can illustrate our point

```python
y_preds = model.predict(X_test)
```

```python
accuracy_score(y_test, y_preds)
```




    0.985



## The problem with JUST accuracy

WOW! won't you look at that, we scored 98.5% on our test set. That's all folks! go home, get on your linked in, call yourself a _dAtA sCiEnTisT_ and start applying for jobs! We crushed it! 

Alas if it were only that easy. 

A key skill you'll need in data science is skepticsim, if the results look too good, they probably are. A Healthy amount of skepticsim will get you far, especially if you start digging into things.

So lets have a look at our actual predictions

```python
Counter(y_preds)
```




    Counter({0.0: 200})



wait a second .. why are we never predicting that a patient has this super rare disease that's life threatening? something feels wrong here. 

Lets look at the definition of accuracy first

Accuracy = $\frac{Number of correct Predictions}{Total number of predictions}$

A quick thought experiment: 

If our model was a child taking a test, whats the best way for this child to cheat on said test if the child was only being graded on accuracy and looking back at past papers over 99% of the answers were 0/False? 

Just Guess everything as 0/False! don't bother studying and spend that free time playing video games

But this isn't a child, its a statistical model and its predicting whether someone has a very rare life threatening illness so we can accept this!

## different types of 'wrong' or errors! 


