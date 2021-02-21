### Model Reuse (MR)


We arbitrarily choose a first video, learn a performance model on it, and select the best configuration minimizing the performance for this video. This approach represents the error made by a model trained on a  source input (i.e., a  first video) and transposed to a target input (i.e., a second video, different from the first one), without considering the difference of content between the source and the target. In theory, it corresponds to a fixed configuration, optimized for the first video. We add Model Reuse as a witness approach to measure how we can improve the standard performance model.

The **Model Reuse** selects a video of the training set, apply a model on it and keep a near-optimal configuration working for this video. Then, it applies this configuration to all inputs of the test set.

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


#### Train set of input videos


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
```




    Index(['subme', 'mixed_ref', 'me_range', 'qpmax', 'aq-mode', 'trellis',
           'fast_pskip', 'chroma_qp_offset', 'bframes', 'ref', 'weightp',
           'rc_lookahead_10', 'rc_lookahead_20', 'rc_lookahead_30',
           'rc_lookahead_40', 'rc_lookahead_50', 'rc_lookahead_60',
           'rc_lookahead_None', 'analyse_0:0', 'analyse_0x113:0x113',
           'analyse_0x3:0x113', 'analyse_0x3:0x133', 'analyse_0x3:0x3', 'me_dia',
           'me_hex', 'me_tesa', 'me_umh', 'b_pyramid_1', 'b_pyramid_2',
           'b_pyramid_None', 'b_adapt_1', 'b_adapt_2', 'b_adapt_None',
           'direct_None', 'direct_auto', 'direct_spatial', 'deblock_0:0:0',
           'deblock_1:0:0', 'weightb_1', 'weightb_None', 'open_gop_0',
           'open_gop_None', 'scenecut_0', 'scenecut_40', 'scenecut_None'],
          dtype='object')



#### Learning Algorithm


```python
mse_lin = []
mse_dt = []
mse_rf = []
mse_bt = []
mse_svr = []
mse_nn = []

for i in range(len(v_names_train[0:50])):
    
    #### Training set/test set of configurations
    X = listVideo[i][cols]
    y = listVideo[i][predDimension]
    
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

    mse_lin.append(mean_squared_error(y_test, ypred_lin))
    mse_dt.append(mean_squared_error(y_test, ypred_dt))
    mse_rf.append(mean_squared_error(y_test, ypred_rf))
    mse_bt.append(mean_squared_error(y_test, y_pred_bt))
    mse_svr.append(mean_squared_error(y_test, y_pred_svr))
    mse_nn.append(mean_squared_error(y_test, y_pred_nn))

print("Average MSE Linear Reg", np.round(np.mean(mse_lin)))
print("Average MSE Decision Tree", np.round(np.mean(mse_dt)))
print("Average MSE Random Forest", np.round(np.mean(mse_rf)))
print("Average MSE Boosting Tree", np.round(np.mean(mse_bt)))
print("Average MSE Support Vector Regressor", np.round(np.mean(mse_svr)))
print("Average MSE Neural Network", np.round(np.mean(mse_nn)))
```

    Average MSE Linear Reg 4.438184404858307e+25
    Average MSE Decision Tree 12701403.0
    Average MSE Random Forest 8200172.0
    Average MSE Boosting Tree 9011004.0
    Average MSE Support Vector Regressor 38002536.0
    Average MSE Neural Network 462307774.0


#### Learning Algorithm kept : Gradient Boosting Trres

#### Hyperparameter optimisation

It is a compromise between the different input videos.


```python
for i in range(10):
    
    #### Training set/test set of configurations
    X = listVideo[i][cols]
    y = listVideo[i][predDimension]
    
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
    [Parallel(n_jobs=5)]: Done  50 tasks      | elapsed:    1.8s
    [Parallel(n_jobs=5)]: Done 396 out of 405 | elapsed:    8.6s remaining:    0.2s
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    8.8s finished


    {'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 100}
    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.
    [Parallel(n_jobs=5)]: Done 110 tasks      | elapsed:    2.1s
    [Parallel(n_jobs=5)]: Done 396 out of 405 | elapsed:    7.5s remaining:    0.2s
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    7.8s finished
    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.


    {'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 100}
    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Done 110 tasks      | elapsed:    2.2s
    [Parallel(n_jobs=5)]: Done 396 out of 405 | elapsed:    7.6s remaining:    0.2s
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    7.8s finished
    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.


    {'max_depth': 5, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 10}
    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Done 110 tasks      | elapsed:    2.3s
    [Parallel(n_jobs=5)]: Done 396 out of 405 | elapsed:    8.1s remaining:    0.2s
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    8.4s finished
    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.


    {'max_depth': None, 'max_features': 15, 'min_samples_leaf': 2, 'n_estimators': 100}
    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Done 110 tasks      | elapsed:    2.6s
    [Parallel(n_jobs=5)]: Done 396 out of 405 | elapsed:    8.0s remaining:    0.2s
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    8.2s finished


    {'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 100}
    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.
    [Parallel(n_jobs=5)]: Done 110 tasks      | elapsed:    2.2s
    [Parallel(n_jobs=5)]: Done 396 out of 405 | elapsed:    7.5s remaining:    0.2s
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    7.7s finished
    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.


    {'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 50}
    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Done 110 tasks      | elapsed:    2.1s
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    7.9s finished
    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.


    {'max_depth': None, 'max_features': 15, 'min_samples_leaf': 2, 'n_estimators': 100}
    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Done 110 tasks      | elapsed:    2.3s
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    8.2s finished
    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.


    {'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 50}
    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Done 110 tasks      | elapsed:    2.5s
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    8.0s finished
    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.


    {'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 50}
    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Done 110 tasks      | elapsed:    2.2s


    {'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 100}


    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed:    7.8s finished


#### We count the number of occurences :

{'max_depth' : None, 'max_features' : 33, 'min_sample_leaf' : 2, 'n_estimators' : 50}

#### Test set of inputs


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
```




    Index(['subme', 'mixed_ref', 'me_range', 'qpmax', 'aq-mode', 'trellis',
           'fast_pskip', 'chroma_qp_offset', 'bframes', 'ref', 'weightp',
           'rc_lookahead_10', 'rc_lookahead_20', 'rc_lookahead_30',
           'rc_lookahead_40', 'rc_lookahead_50', 'rc_lookahead_60',
           'rc_lookahead_None', 'analyse_0:0', 'analyse_0x113:0x113',
           'analyse_0x3:0x113', 'analyse_0x3:0x133', 'analyse_0x3:0x3', 'me_dia',
           'me_hex', 'me_tesa', 'me_umh', 'b_pyramid_1', 'b_pyramid_2',
           'b_pyramid_None', 'b_adapt_1', 'b_adapt_2', 'b_adapt_None',
           'direct_None', 'direct_auto', 'direct_spatial', 'deblock_0:0:0',
           'deblock_1:0:0', 'weightb_1', 'weightb_None', 'open_gop_0',
           'open_gop_None', 'scenecut_0', 'scenecut_40', 'scenecut_None'],
          dtype='object')



#### For each input of the test set, we select arbitrarly an input of the training set, and train a model on it, then we keep this configuration for the test input


```python
best_config_MR = []

for i in range(len(v_names_test)):
    # arbitrarly chosen
    index_train = 788
    vid = listVideo[index_train]
    X = vid[cols]
    y = vid[predDimension]
    
    LA_rf = RandomForestRegressor(max_depth = None, 
                                  max_features = 33, 
                                  min_samples_leaf = 2, 
                                  n_estimators = 50)
    LA_rf.fit(X, y)
    y_pred = LA_rf.predict(X)
    
    best_config_MR.append(np.argmin(y_pred))  
```

#### Print the results


```python
best_config_MR
```




    [175,
     92,
     175,
     193,
     92,
     193,
     175,
     175,
     175,
     193,
     175,
     175,
     193,
     175,
     193,
     175,
     175,
     92,
     175,
     92,
     175,
     175,
     175,
     193,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     175,
     175,
     193,
     175,
     164,
     175,
     175,
     164,
     92,
     92,
     175,
     175,
     193,
     175,
     92,
     175,
     175,
     193,
     164,
     175,
     193,
     175,
     175,
     175,
     92,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     175,
     193,
     175,
     175,
     92,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     175,
     175,
     175,
     175,
     175,
     92,
     175,
     92,
     92,
     193,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     175,
     92,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     193,
     175,
     175,
     193,
     175,
     175,
     175,
     175,
     175,
     193,
     2,
     175,
     175,
     175,
     175,
     193,
     92,
     175,
     175,
     92,
     175,
     175,
     92,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     175,
     193,
     175,
     175,
     175,
     92,
     175,
     175,
     193,
     175,
     175,
     175,
     175,
     92,
     175,
     175,
     175,
     175,
     92,
     175,
     175,
     175,
     175,
     193,
     193,
     175,
     175,
     92,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     164,
     193,
     92,
     175,
     175,
     175,
     92,
     175,
     175,
     193,
     175,
     175,
     175,
     175,
     175,
     175,
     92,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     92,
     193,
     175,
     92,
     193,
     175,
     193,
     175,
     175,
     193,
     175,
     92,
     175,
     175,
     175,
     175,
     164,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     92,
     193,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     164,
     175,
     193,
     175,
     175,
     175,
     175,
     175,
     193,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     175,
     193,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     193,
     92,
     175,
     175,
     92,
     175,
     193,
     175,
     175,
     193,
     175,
     175,
     175,
     175,
     175,
     193,
     193,
     175,
     92,
     193,
     175,
     175,
     175,
     175,
     175,
     175,
     193,
     175,
     164,
     92,
     175,
     175,
     175,
     164,
     92,
     175]



#### Save the results


```python
#np.savetxt("../../../results/raw_data/MR_results.csv", best_config_MR, fmt = '%i')
```


```python

```


```python

```
