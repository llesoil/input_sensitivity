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

import statsmodels.api as sm

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


```python
class MS:
    
    def __init__(self):
        #self.pct_test = pct_test
        #self.ratio_exploitation = ratio_exploitation
        
        # the data folder, see the markdown there for additional explanations
        res_dir = "../../../data/ugc/res_ugc/"
        
        # the list of videos names, e.g. Animation_360P-3e40
        # we sort the list so we keep the same ids between two launches
        v_names = sorted(os.listdir(res_dir)) 

        self.predDimension = "kbs"
        
        # the list of measurements
        listVideo = []

        # we add each dataset in the list, converting the time to the right format
        # third line asserts that the measures are complete
        for v in v_names:
            data = pd.read_table(res_dir+v, delimiter = ',')
            inter = pd.get_dummies(data)
            inter[self.predDimension] = data[self.predDimension]
            listVideo.append(inter)
        
        self.listVideo = listVideo
        
        
        # to sample the source and the target using the same seed
        self.random_state = np.random.randint(0,1000)
        
        self.features = ['cabac', '8x8dct', 'mbtree', 'rc_lookahead', 'analyse', 'me', 'subme', 'mixed_ref', 'me_range', 
                 'qpmax', 'aq-mode', 'trellis','fast_pskip', 'chroma_qp_offset', 'bframes', 'b_pyramid', 
                 'b_adapt', 'direct', 'ref', 'deblock', 'weightb', 'open_gop', 'weightp', 'scenecut']
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)
    
    def learn(self, source_id, target_id, train_size, 
                    learning_algorithm = RandomForestRegressor, 
                    shift_function = RandomForestRegressor):
    
        # the percentage (proportion) of configurations used for the test
        # pct_test = 1-nb_config_target_training/len(listVideo[target_id].index)
        # print(pct_test)

        # random state , i.e. a seed to split the source and the target datasets
        # by using the same set of configurations for training and testing
        random_state = np.random.randint(0,1000)

        # We define the source video, and split it into train-test
        source = self.listVideo[source_id]
        X_src = source.drop([self.predDimension], axis = 1)
        y_src = np.array(source[self.predDimension], dtype=float)
        X_src_train, X_src_test, y_src_train, y_src_test = train_test_split(X_src, 
                                                                            y_src, 
                                                                            train_size=train_size,
                                                                            random_state=random_state)
        #print(X_src_train.shape)
        # We define the target video, and split it into train-test
        target = self.listVideo[target_id]
        X_tgt = target.drop([self.predDimension], axis = 1)
        y_tgt = np.array(target[self.predDimension], dtype=float)
        X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(X_tgt, 
                                                                            y_tgt, 
                                                                            train_size=train_size, 
                                                                            random_state=random_state)

        # The learning algorithm, training on the source video
        X_src_train2, _, y_src_train2, _ = train_test_split(X_src, y_src, 
                                                            test_size=0.7)
        
        lf = learning_algorithm()
        lf.fit(X_src_train2, y_src_train2)
        y_src_pred_test = np.array(lf.predict(X_src_test)).reshape(-1,1)

        # The shift function, to transfer the prediction from the source to the target
        shift = shift_function()
        shift.fit(np.array(y_src_train).reshape(-1,1), y_tgt_train)
        y_tgt_pred_test = shift.predict(y_src.reshape(-1,1))

        # We return the mean average percentage error 
        # between the real values of y_test from target 
        # and the predictions shifted 
        return self.mse(y_tgt_pred_test, y_tgt)
    
    def predict_conf(self, source_id, target_id, train_size, 
                    learning_algorithm = RandomForestRegressor, 
                    shift_function = RandomForestRegressor):
    
        # the percentage (proportion) of configurations used for the test
        # pct_test = 1-nb_config_target_training/len(listVideo[target_id].index)
        # print(pct_test)

        # random state , i.e. a seed to split the source and the target datasets
        # by using the same set of configurations for training and testing
        random_state = np.random.randint(0,1000)

        # We define the source video, and split it into train-test
        source = self.listVideo[source_id]
        X_src = source.drop([self.predDimension], axis = 1)
        y_src = np.array(source[self.predDimension], dtype=float)
        X_src_train, X_src_test, y_src_train, y_src_test = train_test_split(X_src, 
                                                                            y_src, 
                                                                            train_size=train_size,
                                                                            random_state=random_state)
        #print(X_src_train.shape)
        # We define the target video, and split it into train-test
        target = self.listVideo[target_id]
        X_tgt = target.drop([self.predDimension], axis = 1)
        y_tgt = np.array(target[self.predDimension], dtype=float)
        X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(X_tgt, 
                                                                            y_tgt, 
                                                                            train_size=train_size, 
                                                                            random_state=random_state)

        # The learning algorithm, training on the source video
        X_src_train2, _, y_src_train2, _ = train_test_split(X_src, y_src, 
                                                            test_size=0.7)
        
        lf = learning_algorithm()
        lf.fit(X_src_train2, y_src_train2)
        y_src_pred_test = np.array(lf.predict(X_src_test)).reshape(-1,1)

        # The shift function, to transfer the prediction from the source to the target
        shift = shift_function()
        shift.fit(np.array(y_src_train).reshape(-1,1), y_tgt_train)
        y_tgt_pred_test = shift.predict(y_src.reshape(-1,1))

        # We return the mean average percentage error 
        # between the real values of y_test from target 
        # and the predictions shifted 
        return np.argmin(y_tgt_pred_test)
```


```python
ms = MS()

for ts in np.arange(5,31,5):
    print(pct_test, ms.learn(source_id = 2, target_id = 6, train_size=ts))
```

    0.9 1752906.0207466183
    0.9 1180212.96931652
    0.9 1897579.8515903938
    0.9 3533435.359147508
    0.9 1167846.1500647946
    0.9 980115.2062252816


#### Learning algorithm


```python
LAs = [LinearRegression, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, SVR]
for i in range(5):
    source_id = np.random.randint(0,1000)
    target_id = np.random.randint(0,1000)
    for la in LAs:
        print(la, ms.learn(source_id = source_id, target_id = target_id, 
                           train_size=20, learning_algorithm=la))
```

    <class 'sklearn.linear_model._base.LinearRegression'> 1402725.3941713842
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 1593177.9965469814
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 1771066.38976192
    <class 'xgboost.sklearn.XGBRegressor'> 1062298.221570041
    <class 'sklearn.svm._classes.SVR'> 1693598.6880435469
    <class 'sklearn.linear_model._base.LinearRegression'> 477024.9797989845
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 433654.11879539635
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 416416.2724485392
    <class 'xgboost.sklearn.XGBRegressor'> 532741.3921451636
    <class 'sklearn.svm._classes.SVR'> 473014.2127069929
    <class 'sklearn.linear_model._base.LinearRegression'> 11725.935165688516
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 5476.942739595304
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 5515.880780477623
    <class 'xgboost.sklearn.XGBRegressor'> 12031.106470507546
    <class 'sklearn.svm._classes.SVR'> 4631.240308183757
    <class 'sklearn.linear_model._base.LinearRegression'> 5831065.440052986
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 3748507.6036371174
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 9432613.441799976
    <class 'xgboost.sklearn.XGBRegressor'> 8158091.661674964
    <class 'sklearn.svm._classes.SVR'> 8105087.946786626
    <class 'sklearn.linear_model._base.LinearRegression'> 792507.3883593059
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 528300.459563816
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 1049988.836498537
    <class 'xgboost.sklearn.XGBRegressor'> 758515.6105790148
    <class 'sklearn.svm._classes.SVR'> 434725.5152842457


#### Chosen algorithm :  SVR (however it may depends on the choice of videos)

#### Shifting function


```python
LAs = [LinearRegression, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, SVR]
for i in range(5):
    source_id = np.random.randint(0,1000)
    target_id = np.random.randint(0,1000)
    for la in LAs:
        print(la, ms.learn(source_id = source_id, target_id = target_id, 
                           train_size=20, shift_function=la))
```

    <class 'sklearn.linear_model._base.LinearRegression'> 456217.51119152317
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 670142.3770900498
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 554727.6716164041
    <class 'xgboost.sklearn.XGBRegressor'> 628226.4760653581
    <class 'sklearn.svm._classes.SVR'> 4850385.808003911
    <class 'sklearn.linear_model._base.LinearRegression'> 6398.924680545176
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 11170.511540796018
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 5605.233614417313
    <class 'xgboost.sklearn.XGBRegressor'> 16088.985135634259
    <class 'sklearn.svm._classes.SVR'> 8618.201187216728
    <class 'sklearn.linear_model._base.LinearRegression'> 9593810.98738373
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 10670635.162215425
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 4684757.708281321
    <class 'xgboost.sklearn.XGBRegressor'> 3545923.388881826
    <class 'sklearn.svm._classes.SVR'> 24912526.982189137
    <class 'sklearn.linear_model._base.LinearRegression'> 10137321.052144868
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 15581358.360958707
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 15112328.67821009
    <class 'xgboost.sklearn.XGBRegressor'> 11273944.886928478
    <class 'sklearn.svm._classes.SVR'> 73446955.95778656
    <class 'sklearn.linear_model._base.LinearRegression'> 370035.1328324472
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 355625.1197298507
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 406983.7369410331
    <class 'xgboost.sklearn.XGBRegressor'> 495998.83183285844
    <class 'sklearn.svm._classes.SVR'> 2235762.354274517


#### Chosen algorithm  for shifting function:  RandomForestRegressor (however it may depends on the choice of videos)

We predict the configurations for each video of the test set, for 5 configs, 10 configs, ..., 30 configs in the training set.


```python
# the data folder, see the markdown there for additional explanations
res_dir = "../../../data/ugc/res_ugc/"

# the list of videos names, e.g. Animation_360P-3e40
# we sort the list so we keep the same ids between two launches
v_names = sorted(os.listdir(res_dir)) 

v_names_train = np.loadtxt("../../../results/raw_data/train_names.csv", dtype= str)
v_names_test = np.loadtxt("../../../results/raw_data/test_names.csv", dtype= str)
index_train = [i for i in range(len(v_names)) if v_names[i] in v_names_train]
index_test = [i for i in range(len(v_names)) if v_names[i] in v_names_test]

train_sizes = np.arange(5,31,5)
```


```python
ms = MS()
ms_confs = dict()
for i in range(len(index_test)):
    it = index_test[i]
    source_index_train = np.random.randint(0, len(v_names_train))
    source_id = index_train[source_index_train]
    for ts in train_sizes:
        ms_confs[(i, ts)] = ms.predict_conf(source_id = source_id, target_id = it, train_size=ts,
                                      learning_algorithm = SVR, shift_function = RandomForestRegressor)
```


```python
ms_confs
```




    {(0, 5): 8,
     (0, 10): 8,
     (0, 15): 8,
     (0, 20): 67,
     (0, 25): 177,
     (0, 30): 104,
     (1, 5): 8,
     (1, 10): 41,
     (1, 15): 123,
     (1, 20): 168,
     (1, 25): 168,
     (1, 30): 168,
     (2, 5): 4,
     (2, 10): 106,
     (2, 15): 4,
     (2, 20): 32,
     (2, 25): 32,
     (2, 30): 4,
     (3, 5): 1,
     (3, 10): 60,
     (3, 15): 19,
     (3, 20): 104,
     (3, 25): 4,
     (3, 30): 91,
     (4, 5): 0,
     (4, 10): 99,
     (4, 15): 2,
     (4, 20): 6,
     (4, 25): 6,
     (4, 30): 10,
     (5, 5): 2,
     (5, 10): 2,
     (5, 15): 2,
     (5, 20): 9,
     (5, 25): 100,
     (5, 30): 20,
     (6, 5): 1,
     (6, 10): 100,
     (6, 15): 91,
     (6, 20): 105,
     (6, 25): 1,
     (6, 30): 80,
     (7, 5): 0,
     (7, 10): 62,
     (7, 15): 1,
     (7, 20): 2,
     (7, 25): 16,
     (7, 30): 31,
     (8, 5): 1,
     (8, 10): 198,
     (8, 15): 123,
     (8, 20): 123,
     (8, 25): 168,
     (8, 30): 21,
     (9, 5): 1,
     (9, 10): 1,
     (9, 15): 9,
     (9, 20): 73,
     (9, 25): 8,
     (9, 30): 100,
     (10, 5): 5,
     (10, 10): 12,
     (10, 15): 192,
     (10, 20): 4,
     (10, 25): 46,
     (10, 30): 85,
     (11, 5): 7,
     (11, 10): 89,
     (11, 15): 123,
     (11, 20): 179,
     (11, 25): 153,
     (11, 30): 89,
     (12, 5): 1,
     (12, 10): 106,
     (12, 15): 15,
     (12, 20): 25,
     (12, 25): 8,
     (12, 30): 25,
     (13, 5): 26,
     (13, 10): 11,
     (13, 15): 1,
     (13, 20): 62,
     (13, 25): 26,
     (13, 30): 61,
     (14, 5): 4,
     (14, 10): 81,
     (14, 15): 13,
     (14, 20): 78,
     (14, 25): 177,
     (14, 30): 177,
     (15, 5): 6,
     (15, 10): 134,
     (15, 15): 134,
     (15, 20): 123,
     (15, 25): 123,
     (15, 30): 25,
     (16, 5): 2,
     (16, 10): 25,
     (16, 15): 62,
     (16, 20): 24,
     (16, 25): 50,
     (16, 30): 24,
     (17, 5): 74,
     (17, 10): 3,
     (17, 15): 168,
     (17, 20): 197,
     (17, 25): 166,
     (17, 30): 15,
     (18, 5): 9,
     (18, 10): 9,
     (18, 15): 9,
     (18, 20): 62,
     (18, 25): 75,
     (18, 30): 106,
     (19, 5): 11,
     (19, 10): 100,
     (19, 15): 11,
     (19, 20): 100,
     (19, 25): 62,
     (19, 30): 100,
     (20, 5): 0,
     (20, 10): 4,
     (20, 15): 101,
     (20, 20): 32,
     (20, 25): 95,
     (20, 30): 40,
     (21, 5): 1,
     (21, 10): 11,
     (21, 15): 12,
     (21, 20): 97,
     (21, 25): 57,
     (21, 30): 100,
     (22, 5): 1,
     (22, 10): 17,
     (22, 15): 153,
     (22, 20): 1,
     (22, 25): 197,
     (22, 30): 169,
     (23, 5): 1,
     (23, 10): 17,
     (23, 15): 14,
     (23, 20): 4,
     (23, 25): 9,
     (23, 30): 190,
     (24, 5): 10,
     (24, 10): 45,
     (24, 15): 62,
     (24, 20): 93,
     (24, 25): 100,
     (24, 30): 54,
     (25, 5): 2,
     (25, 10): 9,
     (25, 15): 91,
     (25, 20): 60,
     (25, 25): 190,
     (25, 30): 169,
     (26, 5): 8,
     (26, 10): 51,
     (26, 15): 2,
     (26, 20): 100,
     (26, 25): 100,
     (26, 30): 8,
     (27, 5): 32,
     (27, 10): 75,
     (27, 15): 75,
     (27, 20): 32,
     (27, 25): 32,
     (27, 30): 194,
     (28, 5): 1,
     (28, 10): 42,
     (28, 15): 179,
     (28, 20): 42,
     (28, 25): 23,
     (28, 30): 91,
     (29, 5): 123,
     (29, 10): 8,
     (29, 15): 123,
     (29, 20): 123,
     (29, 25): 91,
     (29, 30): 21,
     (30, 5): 2,
     (30, 10): 1,
     (30, 15): 89,
     (30, 20): 137,
     (30, 25): 195,
     (30, 30): 25,
     (31, 5): 0,
     (31, 10): 8,
     (31, 15): 112,
     (31, 20): 29,
     (31, 25): 42,
     (31, 30): 192,
     (32, 5): 2,
     (32, 10): 0,
     (32, 15): 21,
     (32, 20): 46,
     (32, 25): 187,
     (32, 30): 190,
     (33, 5): 2,
     (33, 10): 9,
     (33, 15): 166,
     (33, 20): 2,
     (33, 25): 23,
     (33, 30): 53,
     (34, 5): 8,
     (34, 10): 1,
     (34, 15): 68,
     (34, 20): 97,
     (34, 25): 51,
     (34, 30): 137,
     (35, 5): 10,
     (35, 10): 23,
     (35, 15): 0,
     (35, 20): 7,
     (35, 25): 86,
     (35, 30): 7,
     (36, 5): 1,
     (36, 10): 27,
     (36, 15): 8,
     (36, 20): 179,
     (36, 25): 184,
     (36, 30): 67,
     (37, 5): 29,
     (37, 10): 4,
     (37, 15): 34,
     (37, 20): 0,
     (37, 25): 76,
     (37, 30): 34,
     (38, 5): 2,
     (38, 10): 70,
     (38, 15): 2,
     (38, 20): 16,
     (38, 25): 39,
     (38, 30): 98,
     (39, 5): 1,
     (39, 10): 7,
     (39, 15): 85,
     (39, 20): 20,
     (39, 25): 21,
     (39, 30): 76,
     (40, 5): 5,
     (40, 10): 65,
     (40, 15): 35,
     (40, 20): 47,
     (40, 25): 71,
     (40, 30): 111,
     (41, 5): 20,
     (41, 10): 61,
     (41, 15): 23,
     (41, 20): 62,
     (41, 25): 165,
     (41, 30): 47,
     (42, 5): 13,
     (42, 10): 13,
     (42, 15): 134,
     (42, 20): 8,
     (42, 25): 46,
     (42, 30): 169,
     (43, 5): 1,
     (43, 10): 89,
     (43, 15): 89,
     (43, 20): 62,
     (43, 25): 123,
     (43, 30): 8,
     (44, 5): 2,
     (44, 10): 2,
     (44, 15): 25,
     (44, 20): 123,
     (44, 25): 25,
     (44, 30): 2,
     (45, 5): 10,
     (45, 10): 71,
     (45, 15): 26,
     (45, 20): 39,
     (45, 25): 32,
     (45, 30): 71,
     (46, 5): 104,
     (46, 10): 32,
     (46, 15): 15,
     (46, 20): 46,
     (46, 25): 104,
     (46, 30): 104,
     (47, 5): 123,
     (47, 10): 123,
     (47, 15): 123,
     (47, 20): 166,
     (47, 25): 168,
     (47, 30): 177,
     (48, 5): 1,
     (48, 10): 2,
     (48, 15): 23,
     (48, 20): 15,
     (48, 25): 91,
     (48, 30): 91,
     (49, 5): 39,
     (49, 10): 10,
     (49, 15): 185,
     (49, 20): 88,
     (49, 25): 42,
     (49, 30): 147,
     (50, 5): 8,
     (50, 10): 85,
     (50, 15): 69,
     (50, 20): 75,
     (50, 25): 8,
     (50, 30): 23,
     (51, 5): 1,
     (51, 10): 4,
     (51, 15): 26,
     (51, 20): 46,
     (51, 25): 23,
     (51, 30): 36,
     (52, 5): 2,
     (52, 10): 32,
     (52, 15): 32,
     (52, 20): 2,
     (52, 25): 134,
     (52, 30): 2,
     (53, 5): 8,
     (53, 10): 60,
     (53, 15): 25,
     (53, 20): 62,
     (53, 25): 169,
     (53, 30): 170,
     (54, 5): 21,
     (54, 10): 70,
     (54, 15): 20,
     (54, 20): 10,
     (54, 25): 164,
     (54, 30): 112,
     (55, 5): 2,
     (55, 10): 5,
     (55, 15): 11,
     (55, 20): 4,
     (55, 25): 32,
     (55, 30): 31,
     (56, 5): 25,
     (56, 10): 2,
     (56, 15): 123,
     (56, 20): 74,
     (56, 25): 137,
     (56, 30): 170,
     (57, 5): 9,
     (57, 10): 93,
     (57, 15): 85,
     (57, 20): 94,
     (57, 25): 28,
     (57, 30): 26,
     (58, 5): 62,
     (58, 10): 25,
     (58, 15): 91,
     (58, 20): 13,
     (58, 25): 62,
     (58, 30): 198,
     (59, 5): 67,
     (59, 10): 32,
     (59, 15): 4,
     (59, 20): 8,
     (59, 25): 104,
     (59, 30): 46,
     (60, 5): 1,
     (60, 10): 89,
     (60, 15): 21,
     (60, 20): 123,
     (60, 25): 168,
     (60, 30): 89,
     (61, 5): 5,
     (61, 10): 8,
     (61, 15): 12,
     (61, 20): 34,
     (61, 25): 34,
     (61, 30): 7,
     (62, 5): 95,
     (62, 10): 67,
     (62, 15): 67,
     (62, 20): 89,
     (62, 25): 123,
     (62, 30): 123,
     (63, 5): 4,
     (63, 10): 4,
     (63, 15): 17,
     (63, 20): 104,
     (63, 25): 104,
     (63, 30): 104,
     (64, 5): 1,
     (64, 10): 9,
     (64, 15): 29,
     (64, 20): 7,
     (64, 25): 13,
     (64, 30): 12,
     (65, 5): 1,
     (65, 10): 1,
     (65, 15): 180,
     (65, 20): 62,
     (65, 25): 21,
     (65, 30): 89,
     (66, 5): 4,
     (66, 10): 5,
     (66, 15): 4,
     (66, 20): 61,
     (66, 25): 61,
     (66, 30): 38,
     (67, 5): 1,
     (67, 10): 9,
     (67, 15): 15,
     (67, 20): 2,
     (67, 25): 42,
     (67, 30): 165,
     (68, 5): 2,
     (68, 10): 8,
     (68, 15): 2,
     (68, 20): 177,
     (68, 25): 1,
     (68, 30): 1,
     (69, 5): 80,
     (69, 10): 26,
     (69, 15): 91,
     (69, 20): 80,
     (69, 25): 168,
     (69, 30): 8,
     (70, 5): 2,
     (70, 10): 123,
     (70, 15): 8,
     (70, 20): 113,
     (70, 25): 15,
     (70, 30): 113,
     (71, 5): 74,
     (71, 10): 123,
     (71, 15): 4,
     (71, 20): 4,
     (71, 25): 171,
     (71, 30): 187,
     (72, 5): 34,
     (72, 10): 9,
     (72, 15): 1,
     (72, 20): 5,
     (72, 25): 168,
     (72, 30): 4,
     (73, 5): 1,
     (73, 10): 32,
     (73, 15): 32,
     (73, 20): 34,
     (73, 25): 4,
     (73, 30): 4,
     (74, 5): 1,
     (74, 10): 123,
     (74, 15): 94,
     (74, 20): 75,
     (74, 25): 96,
     (74, 30): 195,
     (75, 5): 164,
     (75, 10): 96,
     (75, 15): 15,
     (75, 20): 2,
     (75, 25): 164,
     (75, 30): 2,
     (76, 5): 1,
     (76, 10): 20,
     (76, 15): 17,
     (76, 20): 21,
     (76, 25): 17,
     (76, 30): 56,
     (77, 5): 1,
     (77, 10): 4,
     (77, 15): 32,
     (77, 20): 32,
     (77, 25): 32,
     (77, 30): 104,
     (78, 5): 4,
     (78, 10): 4,
     (78, 15): 7,
     (78, 20): 56,
     (78, 25): 104,
     (78, 30): 34,
     (79, 5): 0,
     (79, 10): 6,
     (79, 15): 28,
     (79, 20): 26,
     (79, 25): 32,
     (79, 30): 70,
     (80, 5): 193,
     (80, 10): 1,
     (80, 15): 21,
     (80, 20): 21,
     (80, 25): 21,
     (80, 30): 168,
     (81, 5): 174,
     (81, 10): 2,
     (81, 15): 194,
     (81, 20): 123,
     (81, 25): 3,
     (81, 30): 123,
     (82, 5): 20,
     (82, 10): 91,
     (82, 15): 123,
     (82, 20): 123,
     (82, 25): 91,
     (82, 30): 197,
     (83, 5): 4,
     (83, 10): 56,
     (83, 15): 103,
     (83, 20): 56,
     (83, 25): 56,
     (83, 30): 166,
     (84, 5): 62,
     (84, 10): 12,
     (84, 15): 169,
     (84, 20): 180,
     (84, 25): 60,
     (84, 30): 103,
     (85, 5): 100,
     (85, 10): 4,
     (85, 15): 26,
     (85, 20): 54,
     (85, 25): 12,
     (85, 30): 169,
     (86, 5): 4,
     (86, 10): 1,
     (86, 15): 62,
     (86, 20): 83,
     (86, 25): 161,
     (86, 30): 169,
     (87, 5): 4,
     (87, 10): 32,
     (87, 15): 4,
     (87, 20): 32,
     (87, 25): 32,
     (87, 30): 32,
     (88, 5): 8,
     (88, 10): 5,
     (88, 15): 35,
     (88, 20): 57,
     (88, 25): 190,
     (88, 30): 135,
     (89, 5): 2,
     (89, 10): 15,
     (89, 15): 26,
     (89, 20): 67,
     (89, 25): 19,
     (89, 30): 74,
     (90, 5): 1,
     (90, 10): 1,
     (90, 15): 1,
     (90, 20): 62,
     (90, 25): 62,
     (90, 30): 75,
     (91, 5): 8,
     (91, 10): 93,
     (91, 15): 20,
     (91, 20): 91,
     (91, 25): 91,
     (91, 30): 171,
     (92, 5): 1,
     (92, 10): 134,
     (92, 15): 19,
     (92, 20): 134,
     (92, 25): 23,
     (92, 30): 23,
     (93, 5): 9,
     (93, 10): 29,
     (93, 15): 4,
     (93, 20): 9,
     (93, 25): 123,
     (93, 30): 9,
     (94, 5): 123,
     (94, 10): 1,
     (94, 15): 23,
     (94, 20): 8,
     (94, 25): 46,
     (94, 30): 123,
     (95, 5): 160,
     (95, 10): 6,
     (95, 15): 89,
     (95, 20): 123,
     (95, 25): 153,
     (95, 30): 169,
     (96, 5): 1,
     (96, 10): 13,
     (96, 15): 15,
     (96, 20): 192,
     (96, 25): 2,
     (96, 30): 92,
     (97, 5): 1,
     (97, 10): 4,
     (97, 15): 5,
     (97, 20): 4,
     (97, 25): 36,
     (97, 30): 67,
     (98, 5): 9,
     (98, 10): 20,
     (98, 15): 20,
     (98, 20): 1,
     (98, 25): 123,
     (98, 30): 153,
     (99, 5): 10,
     (99, 10): 67,
     (99, 15): 3,
     (99, 20): 3,
     (99, 25): 89,
     (99, 30): 20,
     (100, 5): 2,
     (100, 10): 6,
     (100, 15): 88,
     (100, 20): 92,
     (100, 25): 2,
     (100, 30): 3,
     (101, 5): 2,
     (101, 10): 23,
     (101, 15): 2,
     (101, 20): 3,
     (101, 25): 2,
     (101, 30): 2,
     (102, 5): 81,
     (102, 10): 34,
     (102, 15): 123,
     (102, 20): 165,
     (102, 25): 26,
     (102, 30): 36,
     (103, 5): 13,
     (103, 10): 143,
     (103, 15): 84,
     (103, 20): 165,
     (103, 25): 143,
     (103, 30): 123,
     (104, 5): 20,
     (104, 10): 172,
     (104, 15): 20,
     (104, 20): 162,
     (104, 25): 104,
     (104, 30): 123,
     (105, 5): 13,
     (105, 10): 64,
     (105, 15): 170,
     (105, 20): 123,
     (105, 25): 9,
     (105, 30): 1,
     (106, 5): 2,
     (106, 10): 14,
     (106, 15): 14,
     (106, 20): 174,
     (106, 25): 26,
     (106, 30): 8,
     (107, 5): 2,
     (107, 10): 0,
     (107, 15): 15,
     (107, 20): 88,
     (107, 25): 2,
     (107, 30): 88,
     (108, 5): 1,
     (108, 10): 27,
     (108, 15): 1,
     (108, 20): 61,
     (108, 25): 179,
     (108, 30): 134,
     (109, 5): 6,
     (109, 10): 1,
     (109, 15): 9,
     (109, 20): 1,
     (109, 25): 5,
     (109, 30): 20,
     (110, 5): 21,
     (110, 10): 123,
     (110, 15): 9,
     (110, 20): 169,
     (110, 25): 2,
     (110, 30): 169,
     (111, 5): 0,
     (111, 10): 0,
     (111, 15): 6,
     (111, 20): 186,
     (111, 25): 92,
     (111, 30): 197,
     (112, 5): 89,
     (112, 10): 89,
     (112, 15): 166,
     (112, 20): 123,
     (112, 25): 8,
     (112, 30): 168,
     (113, 5): 29,
     (113, 10): 38,
     (113, 15): 5,
     (113, 20): 60,
     (113, 25): 40,
     (113, 30): 195,
     (114, 5): 6,
     (114, 10): 15,
     (114, 15): 63,
     (114, 20): 76,
     (114, 25): 12,
     (114, 30): 179,
     (115, 5): 20,
     (115, 10): 0,
     (115, 15): 91,
     (115, 20): 91,
     (115, 25): 104,
     (115, 30): 10,
     (116, 5): 1,
     (116, 10): 12,
     (116, 15): 4,
     (116, 20): 4,
     (116, 25): 4,
     (116, 30): 12,
     (117, 5): 2,
     (117, 10): 0,
     (117, 15): 2,
     (117, 20): 3,
     (117, 25): 2,
     (117, 30): 170,
     (118, 5): 3,
     (118, 10): 91,
     (118, 15): 20,
     (118, 20): 20,
     (118, 25): 24,
     (118, 30): 168,
     (119, 5): 4,
     (119, 10): 8,
     (119, 15): 20,
     (119, 20): 123,
     (119, 25): 15,
     (119, 30): 170,
     (120, 5): 6,
     (120, 10): 84,
     (120, 15): 140,
     (120, 20): 31,
     (120, 25): 1,
     (120, 30): 150,
     (121, 5): 4,
     (121, 10): 3,
     (121, 15): 15,
     (121, 20): 165,
     (121, 25): 136,
     (121, 30): 112,
     (122, 5): 1,
     (122, 10): 47,
     (122, 15): 60,
     (122, 20): 92,
     (122, 25): 75,
     (122, 30): 80,
     (123, 5): 52,
     (123, 10): 40,
     (123, 15): 9,
     (123, 20): 107,
     (123, 25): 170,
     (123, 30): 40,
     (124, 5): 9,
     (124, 10): 4,
     (124, 15): 73,
     (124, 20): 4,
     (124, 25): 95,
     (124, 30): 20,
     (125, 5): 123,
     (125, 10): 89,
     (125, 15): 89,
     (125, 20): 20,
     (125, 25): 123,
     (125, 30): 123,
     (126, 5): 8,
     (126, 10): 195,
     (126, 15): 50,
     (126, 20): 195,
     (126, 25): 62,
     (126, 30): 20,
     (127, 5): 104,
     (127, 10): 1,
     (127, 15): 12,
     (127, 20): 38,
     (127, 25): 5,
     (127, 30): 123,
     (128, 5): 165,
     (128, 10): 20,
     (128, 15): 104,
     (128, 20): 104,
     (128, 25): 165,
     (128, 30): 165,
     (129, 5): 2,
     (129, 10): 7,
     (129, 15): 169,
     (129, 20): 171,
     (129, 25): 168,
     (129, 30): 165,
     (130, 5): 135,
     (130, 10): 41,
     (130, 15): 20,
     (130, 20): 65,
     (130, 25): 123,
     (130, 30): 26,
     (131, 5): 123,
     (131, 10): 3,
     (131, 15): 123,
     (131, 20): 5,
     (131, 25): 123,
     (131, 30): 123,
     (132, 5): 2,
     (132, 10): 2,
     (132, 15): 154,
     (132, 20): 26,
     (132, 25): 2,
     (132, 30): 179,
     (133, 5): 2,
     (133, 10): 32,
     (133, 15): 2,
     (133, 20): 23,
     (133, 25): 60,
     (133, 30): 41,
     (134, 5): 1,
     (134, 10): 196,
     (134, 15): 134,
     (134, 20): 112,
     (134, 25): 18,
     (134, 30): 2,
     (135, 5): 177,
     (135, 10): 28,
     (135, 15): 29,
     (135, 20): 75,
     (135, 25): 94,
     (135, 30): 8,
     (136, 5): 1,
     (136, 10): 15,
     (136, 15): 14,
     (136, 20): 15,
     (136, 25): 106,
     (136, 30): 29,
     (137, 5): 0,
     (137, 10): 26,
     (137, 15): 75,
     (137, 20): 28,
     (137, 25): 168,
     (137, 30): 10,
     (138, 5): 4,
     (138, 10): 60,
     (138, 15): 68,
     (138, 20): 106,
     (138, 25): 46,
     (138, 30): 36,
     (139, 5): 0,
     (139, 10): 5,
     (139, 15): 29,
     (139, 20): 2,
     (139, 25): 47,
     (139, 30): 47,
     (140, 5): 1,
     (140, 10): 5,
     (140, 15): 196,
     (140, 20): 92,
     (140, 25): 12,
     (140, 30): 130,
     (141, 5): 1,
     (141, 10): 63,
     (141, 15): 6,
     (141, 20): 90,
     (141, 25): 164,
     (141, 30): 60,
     (142, 5): 4,
     (142, 10): 12,
     (142, 15): 4,
     (142, 20): 4,
     (142, 25): 15,
     (142, 30): 166,
     (143, 5): 1,
     (143, 10): 1,
     (143, 15): 15,
     (143, 20): 123,
     (143, 25): 170,
     (143, 30): 153,
     (144, 5): 8,
     (144, 10): 38,
     (144, 15): 38,
     (144, 20): 171,
     (144, 25): 95,
     (144, 30): 165,
     (145, 5): 4,
     (145, 10): 2,
     (145, 15): 2,
     (145, 20): 123,
     (145, 25): 169,
     (145, 30): 166,
     (146, 5): 8,
     (146, 10): 85,
     (146, 15): 48,
     (146, 20): 48,
     (146, 25): 48,
     (146, 30): 76,
     (147, 5): 8,
     (147, 10): 123,
     (147, 15): 123,
     (147, 20): 123,
     (147, 25): 123,
     (147, 30): 197,
     (148, 5): 10,
     (148, 10): 2,
     (148, 15): 2,
     (148, 20): 24,
     (148, 25): 166,
     (148, 30): 21,
     (149, 5): 4,
     (149, 10): 17,
     (149, 15): 2,
     (149, 20): 2,
     (149, 25): 45,
     (149, 30): 23,
     (150, 5): 8,
     (150, 10): 1,
     (150, 15): 80,
     (150, 20): 107,
     (150, 25): 65,
     (150, 30): 65,
     (151, 5): 1,
     (151, 10): 7,
     (151, 15): 7,
     (151, 20): 27,
     (151, 25): 67,
     (151, 30): 6,
     (152, 5): 5,
     (152, 10): 15,
     (152, 15): 179,
     (152, 20): 26,
     (152, 25): 165,
     (152, 30): 179,
     (153, 5): 0,
     (153, 10): 0,
     (153, 15): 70,
     (153, 20): 5,
     (153, 25): 71,
     (153, 30): 35,
     (154, 5): 161,
     (154, 10): 39,
     (154, 15): 7,
     (154, 20): 91,
     (154, 25): 28,
     (154, 30): 51,
     (155, 5): 0,
     (155, 10): 12,
     (155, 15): 12,
     (155, 20): 12,
     (155, 25): 32,
     (155, 30): 4,
     (156, 5): 11,
     (156, 10): 2,
     (156, 15): 153,
     (156, 20): 53,
     (156, 25): 153,
     (156, 30): 153,
     (157, 5): 1,
     (157, 10): 2,
     (157, 15): 2,
     (157, 20): 166,
     (157, 25): 166,
     (157, 30): 169,
     (158, 5): 89,
     (158, 10): 89,
     (158, 15): 21,
     (158, 20): 62,
     (158, 25): 89,
     (158, 30): 9,
     (159, 5): 8,
     (159, 10): 67,
     (159, 15): 69,
     (159, 20): 123,
     (159, 25): 63,
     (159, 30): 166,
     (160, 5): 9,
     (160, 10): 41,
     (160, 15): 9,
     (160, 20): 89,
     (160, 25): 10,
     (160, 30): 196,
     (161, 5): 1,
     (161, 10): 4,
     (161, 15): 62,
     (161, 20): 15,
     (161, 25): 176,
     (161, 30): 155,
     (162, 5): 4,
     (162, 10): 17,
     (162, 15): 123,
     (162, 20): 21,
     (162, 25): 123,
     (162, 30): 123,
     (163, 5): 9,
     (163, 10): 4,
     (163, 15): 4,
     (163, 20): 9,
     (163, 25): 5,
     (163, 30): 26,
     (164, 5): 75,
     (164, 10): 7,
     (164, 15): 67,
     (164, 20): 75,
     (164, 25): 118,
     (164, 30): 60,
     (165, 5): 8,
     (165, 10): 21,
     (165, 15): 8,
     (165, 20): 20,
     (165, 25): 165,
     (165, 30): 89,
     (166, 5): 1,
     (166, 10): 19,
     (166, 15): 123,
     (166, 20): 19,
     ...}




```python
ms_data = pd.DataFrame({"id_video" : [i for i in range(len(index_test))]})
for ts in train_sizes:
    ms_data["conf"+str(ts)] = [ms_confs[(i, ts)] for i in range(len(index_test))]
```


```python
# ms_data.set_index("id_video").to_csv("../../../results/raw_data/MS_results.csv")
```


```python

```
