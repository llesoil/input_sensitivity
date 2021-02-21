### Best compromise (BC)

**Best compromise (BC)** applies a performance model on all the training set, without making a difference between input videos. 
It selects the configuration working best for most videos in the training set. 
Technically, we rank the 201 configurations (1 being the optimal configuration, and 201 the worst) and select the one optimizing the sum of ranks for all input videos in the training set. 

#### Libraries


```python
# for arrays
import numpy as np

# for dataframes
import pandas as pd

# plots
import matplotlib.pyplot as plt
# high-level plots
import seaborn as sns

# statistics
import scipy.stats as sc
# hierarchical clustering, clusters
from scipy.cluster.hierarchy import linkage, cut_tree, leaves_list
from scipy import stats
# statistical tests
from scipy.stats import mannwhitneyu

# machine learning library
# Principal Component Analysis - determine new axis for representing data
from sklearn.decomposition import PCA
# Random Forests -> vote between decision trees
# Gradient boosting -> instead of a vote, upgrade the same tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
# To add interactions in linear regressions models
from sklearn.preprocessing import PolynomialFeatures
# Elasticnet is an hybrid method between ridge and Lasso
from sklearn.linear_model import LinearRegression, ElasticNet
# To separate the data into training and test
from sklearn.model_selection import train_test_split, GridSearchCV
# Simple clustering (iterative steps)
from sklearn.cluster import KMeans
# Support vector machine - support vector regressor
from sklearn.svm import SVR
# decision trees
from sklearn.tree import DecisionTreeRegressor, plot_tree
# mean squared error
from sklearn.metrics import mean_squared_error

# gradient boosting trees
from xgboost import XGBRegressor

# we use it to interact with the file system
import os
# compute time
from time import time

# Neural network high level framework
import keras
# Sequential is a sequence of blocs
# Input deals with the data fed to the network
from keras.models import Sequential,Input,Model
# Dense is a feedforward layer with fully connected nodes
# Dropout allows to keep part of data, and to "drop out" a the rest
# Flatten makes the data "flat", i.e. in one dimension
from keras.layers import Dense, Dropout, Flatten
# Conv -> convolution, MaxPooling is relative to Pooling
# Activation if the function composing the data in output of a layer
from keras.layers import Conv2D, MaxPooling2D, Activation
```

    Using TensorFlow backend.


#### Train set of input videos - Join all the datasets 


```python
v_names_train = np.loadtxt("../../../results/raw_data/train_names.csv", dtype= str)

predDimension = 'kbs'

#because x264 output is "m:s", where m is the number of minutes and s the number of seconds 
# we define a function to convert this format into the number of seconds
def elapsedtime_to_sec(el):
    tab = el.split(":")
    return float(tab[0])*60+float(tab[1])

# the data folder, see the markdown there for additional explanations
res_dir = "../../../data/ugc/res_ugc/"

# the list of videos names, e.g. Animation_360P-3e40
# we sort the list so we keep the same ids between two launches
v_names = sorted(os.listdir(res_dir)) 

to_dummy_features = [ 'rc_lookahead', 'analyse', 'me', 'subme', 'mixed_ref', 'me_range', 'qpmax', 
                      'aq-mode','trellis','fast_pskip', 'chroma_qp_offset', 'bframes', 'b_pyramid', 
                      'b_adapt', 'direct', 'ref', 'deblock', 'weightb', 'open_gop', 'weightp', 
                      'scenecut']

# the list of measurements
listVideo = []

# we add each dataset in the list, converting the time to the right format
# third line asserts that the measures are complete
for v in v_names_train:
    data = pd.read_table(res_dir+v, delimiter = ',')
    data['etime'] = [*map(elapsedtime_to_sec, data['elapsedtime'])]
    assert data.shape == (201,34), v
    inter = pd.get_dummies(data[to_dummy_features])
    inter[predDimension] = data[predDimension]
    listVideo.append(inter)

cols = inter.columns
cols = cols[:len(cols)-1]
cols

final = pd.concat(listVideo, axis=0)
```

#### Learning Algorithm


```python
#### Training set/test set of configurations
X = final[cols]
y = final[predDimension]

# train - test separation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

# Linear regression
lin = LinearRegression()
lin.fit(X_train, y_train)
ypred_lin = lin.predict(X_test)

# Decision Tree
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
ypred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
ypred_rf = rf.predict(X_test)

# Gradient Boosting trees
bt = XGBRegressor()
bt.fit(X_train, y_train)
y_pred_bt = bt.predict(X_test)

# Support Vector Regressor
svr = SVR()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)


# neural network
model_nn = Sequential()
model_nn.add(Dense(45, input_dim=45))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10, activation='relu'))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5, activation='relu'))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(1))

model_nn.compile(loss='MSE', optimizer='Adam')
model_nn.fit(X_train, y_train, epochs=5, verbose = False)
y_pred_nn = model_nn.predict(X_test)

print("Average MSE Linear Reg", np.round(mean_squared_error(y_test, ypred_lin)))
print("Average MSE Decision Tree", np.round(mean_squared_error(y_test, ypred_dt)))
print("Average MSE Random Forest", np.round(mean_squared_error(y_test, ypred_rf)))
print("Average MSE Boosting Tree", np.round(mean_squared_error(y_test, y_pred_bt)))
print("Average MSE Support Vector Regressor", np.round(mean_squared_error(y_test, y_pred_svr)))
print("Average MSE Neural Network", np.round(mean_squared_error(y_test, y_pred_nn)))
```

    Average MSE Linear Reg 261017829.0
    Average MSE Decision Tree 260758459.0
    Average MSE Random Forest 260760034.0
    Average MSE Boosting Tree 260755085.0
    Average MSE Support Vector Regressor 302107392.0
    Average MSE Neural Network 261322284.0


#### Learning Algorithm kept : Random Forest

#### Hyperparameter optimisation

It is a compromise between the different input videos.


```python
#### Training set/test set of configurations
X = final[cols]
y = final[predDimension]

# train - test separation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

LA_rf = RandomForestRegressor()

grid_search_larf = GridSearchCV(estimator = LA_rf,
                                param_grid = {'n_estimators': [10, 50, 100],
                                              # we didn't include 1 for min_samples_leaf to avoid overfitting
                                         'min_samples_leaf' : [2, 5, 10],
                                         'max_depth' : [3, 5, None],
                                         'max_features' : [5, 15, 33]},
                                scoring = 'neg_mean_squared_error',
                                verbose = True,
                                n_jobs = 5)

grid_search_larf.fit(X_train, y_train)

print(grid_search_larf.best_params_)
```

    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.
    [Parallel(n_jobs=5)]: Done  40 tasks      | elapsed:   12.5s
    [Parallel(n_jobs=5)]: Done 190 tasks      | elapsed:  2.4min
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed: 10.7min finished


    {'max_depth': 5, 'max_features': 15, 'min_samples_leaf': 5, 'n_estimators': 100}


#### We count the number of occurences :

{'max_depth': 5, 'max_features': 15, 'min_samples_leaf': 5, 'n_estimators': 100}

#### Test set of input videos


```python
v_names_test = np.loadtxt("../../../results/raw_data/test_names.csv", dtype= str)

# the list of measurements
listVideoTest = []

# we add each dataset in the list, converting the time to the right format
# third line asserts that the measures are complete
for v in v_names_test:
    data = pd.read_table(res_dir+v, delimiter = ',')
    data['etime'] = [*map(elapsedtime_to_sec, data['elapsedtime'])]
    assert data.shape == (201,34), v
    inter = pd.get_dummies(data[to_dummy_features])
    inter[predDimension] = data[predDimension]
    listVideoTest.append(inter)

cols = inter.columns
cols = cols[:len(cols)-1]
cols

final_test = pd.concat(listVideoTest, axis=0)
```

#### Predict results


```python
LA_rf = RandomForestRegressor(max_depth = 5, 
                              max_features = 15, 
                              min_samples_leaf = 5, 
                              n_estimators = 100)

X = final[cols]
y = final[predDimension]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5)

X_te = final_test[cols]
y_te = final_test[predDimension]

LA_rf.fit(X_train, y_train)

y_pred = LA_rf.predict(X_te)
```


```python
assert len(y_pred) == len(v_names_test)*201
```


```python
best_configs_BC = []

for i in range(len(v_names_test)):
    actual_video = [y_pred[i*201+k] for k in range(201)]
    best_configs_BC.append(np.argmin(actual_video))
```


```python
best_configs_BC
```




    [132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132,
     132]



#### Save the results


```python
#np.savetxt("../../../results/raw_data/BC_results.csv", best_configs_BC, fmt = '%i')
```


```python

```
