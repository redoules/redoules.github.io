Title: Article Recommander
Date: 2017-06-23 08:22  
Category: Machine Learning  
Tags: Basics
Slug: Source code for the recommandation engine for articles
Authors: Guillaume Redoulès  
Summary: Source code for the recommandation engine for articles



```python
import pandas as pd
import numpy as np
%matplotlib inline 
```

## Loading data and preprocessing

we first learn the pickled article database. We will be cleaning it and separating the interesting articles from the uninteresting ones. 


```python
df = pd.read_pickle('./article.pkl')
del df["html"]
del df["image"]
del df["URL"]
del df["hash"]
del df["source"]

df["label"] = df["note"].apply(lambda x: 0 if x <= 0 else 1)
df.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>authors</th>
      <th>note</th>
      <th>resume</th>
      <th>texte</th>
      <th>titre</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Danny Bradbury, Marco Santori, Adam Draper, M...</td>
      <td>-10.0</td>
      <td>Black Market Reloaded, a black market site tha...</td>
      <td>Black Market Reloaded, a black market site tha...</td>
      <td>Black Market Reloaded back online after source...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[Emily Spaven, Stan Higgins, Emilyspaven]</td>
      <td>1.0</td>
      <td>The UK Home Office believes the government sho...</td>
      <td>The UK Home Office believes the government sho...</td>
      <td>Home Office: UK Should Create a Crime-Fighting...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[Pete Rizzo, Alex Batlin, Yessi Bello Perez, P...</td>
      <td>-10.0</td>
      <td>Though lofty in its ideals, lead developer Dan...</td>
      <td>A new social messaging app is aiming to disrup...</td>
      <td>Gems Bitcoin App Lets Users Earn Money From So...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[Nermin Hajdarbegovic, Stan Higgins, Pete Rizz...</td>
      <td>3.0</td>
      <td>US satellite service provider DISH Network has...</td>
      <td>US satellite service provider DISH Network has...</td>
      <td>DISH Becomes World's Largest Company to Accept...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[Stan Higgins, Bailey Reutzel, Garrett Keirns,...</td>
      <td>-10.0</td>
      <td>An unidentified 28-year-old man was robbed of ...</td>
      <td>An unidentified 28-year-old man was robbed of ...</td>
      <td>Bitcoin Stolen at Gunpoint in New York City Ro...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Basic statistics on the dataset

let's explore the dataset and extract some numbers : 
* the number of article liked/disliked



```python
df["label"].value_counts()
```




    0    879
    1    324
    Name: label, dtype: int64



## Create the full content column



```python
df['full_content'] = df.titre + ' ' + df.resume  #exclude the full texte of the article for the moment
df.head(1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>authors</th>
      <th>note</th>
      <th>resume</th>
      <th>texte</th>
      <th>titre</th>
      <th>label</th>
      <th>full_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[Danny Bradbury, Marco Santori, Adam Draper, M...</td>
      <td>-10.0</td>
      <td>Black Market Reloaded, a black market site tha...</td>
      <td>Black Market Reloaded, a black market site tha...</td>
      <td>Black Market Reloaded back online after source...</td>
      <td>0</td>
      <td>Black Market Reloaded back online after source...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split
training, testing = train_test_split(
    df,                # The dataset we want to split
    train_size=0.75,    # The proportional size of our training set
    stratify=df.label, # The labels are used for stratification
    random_state=400   # Use the same random state for reproducibility
)

training.head(5)

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>authors</th>
      <th>note</th>
      <th>resume</th>
      <th>texte</th>
      <th>titre</th>
      <th>label</th>
      <th>full_content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>748</th>
      <td>[Jon Brodkin]</td>
      <td>-10.0</td>
      <td>Amazon, Reddit, Mozilla, and other Internet co...</td>
      <td>Amazon, Reddit, Mozilla, and other Internet co...</td>
      <td>Amazon and Reddit try to save net neutrality r...</td>
      <td>0</td>
      <td>Amazon and Reddit try to save net neutrality r...</td>
    </tr>
    <tr>
      <th>1183</th>
      <td>[Jon Brodkin]</td>
      <td>-10.0</td>
      <td>(The Time Warner involved in this transaction ...</td>
      <td>A group of mostly Democratic senators led by A...</td>
      <td>Democrats urge Trump administration to block A...</td>
      <td>0</td>
      <td>Democrats urge Trump administration to block A...</td>
    </tr>
    <tr>
      <th>769</th>
      <td>[Joseph Brogan]</td>
      <td>-10.0</td>
      <td>On Twitter, bad news comes at all hours, with ...</td>
      <td>On Twitter, bad news comes at all hours, with ...</td>
      <td>Some of the best art on Twitter comes from the...</td>
      <td>0</td>
      <td>Some of the best art on Twitter comes from the...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>[Michael Del Castillo, Pete Rizzo, Trond Vidar...</td>
      <td>-10.0</td>
      <td>Publicly traded online travel service Webjet i...</td>
      <td>Publicly traded online travel service Webjet i...</td>
      <td>Webjet Ethereum Pilot Targets Hotel Industry's...</td>
      <td>0</td>
      <td>Webjet Ethereum Pilot Targets Hotel Industry's...</td>
    </tr>
    <tr>
      <th>892</th>
      <td>[Andrew Cunningham]</td>
      <td>10.0</td>
      <td>What has changed on the 2017 MacBook, then?\nI...</td>
      <td>Andrew Cunningham\n\nAndrew Cunningham\n\nAndr...</td>
      <td>Mini-review: The 2017 MacBook could actually b...</td>
      <td>1</td>
      <td>Mini-review: The 2017 MacBook could actually b...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from utils.plotting import pipeline_performance

steps = (
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LinearSVC())
)
pipeline = Pipeline(steps)

predicted_labels = cross_val_predict(pipeline, training.full_content, training.label)
pipeline_performance(training.label, predicted_labels)

pipeline = pipeline.fit(training.titre, training.label)
```

    Accuracy = 80.6%
    Confusion matrix, without normalization
    [[624  35]
     [140 103]]
    


![png]({filename}/images/recommender/output_8_1.png)



```python
import re
from utils.plotting import print_top_features
from sklearn.model_selection import GridSearchCV

def mask_integers(s):
    return re.sub(r'\d+', 'INTMASK', s)
```


```python
steps = (
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LinearSVC())
)

pipeline = Pipeline(steps)

gs_params = {
    #'vectorizer__use_idf': (True, False),
    'vectorizer__lowercase': [True, False],
    'vectorizer__stop_words': ['english', None],
    'vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'vectorizer__preprocessor': [mask_integers, None],
    'classifier__C': np.linspace(5,20,25)
}


gs = GridSearchCV(pipeline, gs_params, n_jobs=1)
gs.fit(training.full_content, training.label)

print(gs.best_params_)
print(gs.best_score_)

pipeline1 = gs.best_estimator_
predicted_labels = pipeline1.predict(testing.full_content)
pipeline_performance(testing.label, predicted_labels)

print_top_features(pipeline1, n_features=10)
```


```python
aaa = gs.predict(testing.full_content) == testing.label 

aaa =  aaa[testing.label == 1]

testing["titre"].iloc[~aaa.values]

#pipeline1.predict(["windows xbox bitcoin"])
from sklearn.externals import joblib
joblib.dump(pipeline1, 'classifier.pkl') 
```


```python
gs.predict(['Google'])

```




    array([1], dtype=int64)




```python
steps = (
    ('vectorizer', TfidfVectorizer()),
    ('classifier', SVC())
)

pipeline = Pipeline(steps)

gs_params = {
    #'vectorizer__use_idf': (True, False),
    'vectorizer__stop_words': ['english', None],
    'vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'vectorizer__preprocessor': [mask_integers, None],
    'classifier__C': np.linspace(5,20,25)
}


gs = GridSearchCV(pipeline, gs_params, n_jobs=1)
gs.fit(training.full_content, training.label)

print(gs.best_params_)
print(gs.best_score_)

pipeline1 = gs.best_estimator_
predicted_labels = pipeline1.predict(testing.full_content)
pipeline_performance(testing.label, predicted_labels)

print_top_features(pipeline1, n_features=10)
```

    {'classifier__C': 5.0, 'vectorizer__ngram_range': (1, 1), 'vectorizer__preprocessor': <function mask_integers at 0x00000237491B67B8>, 'vectorizer__stop_words': 'english'}
    0.711180124224
    Accuracy = 71.2%
    Confusion matrix, without normalization
    [[153   0]
     [ 62   0]]
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-9-3e0781e307fb> in <module>()
         25 pipeline_performance(testing.label, predicted_labels)
         26 
    ---> 27 print_top_features(pipeline1, n_features=10)
    

    C:\Users\Guillaume\Documents\Code\recommandation\utils\plotting.py in print_top_features(pipeline, vectorizer_name, classifier_name, n_features)
         81 def print_top_features(pipeline, vectorizer_name='vectorizer', classifier_name='classifier', n_features=7):
         82     vocabulary = np.array(pipeline.named_steps[vectorizer_name].get_feature_names())
    ---> 83     coefs = pipeline.named_steps[classifier_name].coef_[0]
         84     top_feature_idx = np.argsort(coefs)
         85     top_features = vocabulary[top_feature_idx]
    

    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\svm\base.py in coef_(self)
        483     def coef_(self):
        484         if self.kernel != 'linear':
    --> 485             raise ValueError('coef_ is only available when using a '
        486                              'linear kernel')
        487 
    

    ValueError: coef_ is only available when using a linear kernel



![png]({filename}/images/recommender/output_13_2.png)



```python
from sklearn.naive_bayes import BernoulliNB


steps = (
    ('vectorizer', TfidfVectorizer()),
    ('classifier', BernoulliNB())
)

pipeline2 = Pipeline(steps)

gs_params = {
    'vectorizer__stop_words': ['english', None],
    'vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'vectorizer__preprocessor': [mask_integers, None],
    'classifier__alpha': np.linspace(0,1,5),
    'classifier__fit_prior': [True, False]
}

gs = GridSearchCV(pipeline2, gs_params, n_jobs=1)
gs.fit(training.full_content, training.label)

print(gs.best_params_)
print(gs.best_score_)

pipeline2 = gs.best_estimator_
predicted_labels = pipeline2.predict(testing.full_content)
pipeline_performance(testing.label, predicted_labels)

print_top_features(pipeline2, n_features=10)
```

    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:820: RuntimeWarning: divide by zero encountered in log
      neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:823: RuntimeWarning: invalid value encountered in add
      jll += self.class_log_prior_ + neg_prob.sum(axis=1)
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:801: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    

    {'classifier__alpha': 0.25, 'classifier__fit_prior': True, 'vectorizer__ngram_range': (1, 1), 'vectorizer__preprocessor': <function mask_integers at 0x00000237491B67B8>, 'vectorizer__stop_words': 'english'}
    0.805900621118
    Accuracy = 78.1%
    Confusion matrix, without normalization
    [[140  13]
     [ 34  28]]
    Top like features:
    ['use' 'just' 'year' 'price' 'time' 'Bitcoin' 'bitcoin' 'new' 'The'
     'INTMASK']
    ---
    Top dislike features:
    ['ABBA' 'cable' 'cab' 'byte' 'publication' 'bye' 'publications' 'publicity'
     'buyer' 'publicizing']
    


![png]({filename}/images/recommender/output_14_2.png)



```python
from sklearn.naive_bayes import MultinomialNB


steps = (
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
)

pipeline3 = Pipeline(steps)

gs_params = {
    'vectorizer__stop_words': ['english', None],
    'vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'vectorizer__preprocessor': [mask_integers, None],
    'classifier__alpha': np.linspace(0,1,5),
    'classifier__fit_prior': [True, False]
}

gs = GridSearchCV(pipeline3, gs_params, n_jobs=1)
gs.fit(training.full_content, training.label)

print(gs.best_params_)
print(gs.best_score_)

pipeline3 = gs.best_estimator_
predicted_labels = pipeline3.predict(testing.full_content)
pipeline_performance(testing.label, predicted_labels)

print_top_features(pipeline3, n_features=10)
```

    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    C:\Users\Guillaume\Anaconda3\lib\site-packages\sklearn\naive_bayes.py:699: RuntimeWarning: divide by zero encountered in log
      self.feature_log_prob_ = (np.log(smoothed_fc) -
    

    {'classifier__alpha': 0.5, 'classifier__fit_prior': False, 'vectorizer__ngram_range': (1, 1), 'vectorizer__preprocessor': <function mask_integers at 0x00000237491B67B8>, 'vectorizer__stop_words': 'english'}
    0.80900621118
    Accuracy = 79.1%
    Confusion matrix, without normalization
    [[141  12]
     [ 33  29]]
    Top like features:
    ['time' 'Google' 'Pro' 'Apple' 'new' 'The' 'Bitcoin' 'price' 'bitcoin'
     'INTMASK']
    ---
    Top dislike features:
    ['ABBA' 'categories' 'catching' 'catalyst' 'catalog' 'casually' 'casts'
     'cast' 'cashier' 'ran']
    


![png]({filename}/images/recommender/output_15_2.png)

