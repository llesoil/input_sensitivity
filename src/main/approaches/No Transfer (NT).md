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



```python
class NT:
    
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
    
    def learn(self, target_id, train_size, 
                    learning_algorithm = RandomForestRegressor):
        
        # random state , i.e. a seed to split the source and the target datasets
        # by using the same set of configurations for training and testing
        random_state = np.random.randint(0,1000)
        
        #print(X_src_train.shape)
        # We define the target video, and split it into train-test
        target = self.listVideo[target_id]
        X_tgt = target.drop([self.predDimension], axis = 1)
        y_tgt = np.array(target[self.predDimension], dtype=float)
        X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(X_tgt, 
                                                                            y_tgt, 
                                                                            train_size=train_size, 
                                                                            random_state=random_state)
        
        lf = learning_algorithm()
        lf.fit(X_tgt_train, y_tgt_train)
        y_tgt_pred_test = np.array(lf.predict(X_tgt)).reshape(-1,1)

        # We return the mean average percentage error 
        # between the real values of y_test from target 
        # and the predictions shifted 
        return self.mse(y_tgt_pred_test, y_tgt)
    
    def predict_conf(self, target_id, train_size, 
                    learning_algorithm = RandomForestRegressor):
    
        # the percentage (proportion) of configurations used for the test
        # pct_test = 1-nb_config_target_training/len(listVideo[target_id].index)
        # print(pct_test)

        # random state , i.e. a seed to split the source and the target datasets
        # by using the same set of configurations for training and testing
        random_state = np.random.randint(0,1000)
        
        #print(X_src_train.shape)
        # We define the target video, and split it into train-test
        target = self.listVideo[target_id]
        X_tgt = target.drop([self.predDimension], axis = 1)
        y_tgt = np.array(target[self.predDimension], dtype=float)
        X_tgt_train, X_tgt_test, y_tgt_train, y_tgt_test = train_test_split(X_tgt, 
                                                                            y_tgt, 
                                                                            train_size=train_size, 
                                                                            random_state=random_state)
        
        lf = learning_algorithm()
        lf.fit(X_tgt_train, y_tgt_train)
        y_tgt_pred_test = np.array(lf.predict(X_tgt)).reshape(-1,1)

        # We return the mean average percentage error 
        # between the real values of y_test from target 
        # and the predictions shifted 

        # We return the mean average percentage error 
        # between the real values of y_test from target 
        # and the predictions shifted 
        return np.argmin(y_tgt_pred_test)
```


```python
nt = NT()

for ts in np.arange(5,31,5):
    print(ts, nt.learn(target_id = 6, train_size=ts))
```

    5 5536492.023629223
    10 4797411.582160962
    15 7053105.798687522
    20 5718080.166873196
    25 7325851.463875565
    30 7357854.869430127


#### Learning algorithm


```python
LAs = [LinearRegression, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, SVR]
for i in range(5):
    target_id = np.random.randint(0,1000)
    for la in LAs:
        print(la, nt.learn(target_id = target_id, 
                           train_size=20, learning_algorithm=la))
```

    <class 'sklearn.linear_model._base.LinearRegression'> 90636.724876731
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 96131.63230777207
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 68576.87464658367
    <class 'xgboost.sklearn.XGBRegressor'> 100141.08316194612
    <class 'sklearn.svm._classes.SVR'> 50436.971072850356
    <class 'sklearn.linear_model._base.LinearRegression'> 122000.82175214507
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 89863.02603694216
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 111890.67569140258
    <class 'xgboost.sklearn.XGBRegressor'> 126003.63378314463
    <class 'sklearn.svm._classes.SVR'> 67815.88225900207
    <class 'sklearn.linear_model._base.LinearRegression'> 58610860.38614221
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 59860116.173036136
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 39829093.04593233
    <class 'xgboost.sklearn.XGBRegressor'> 68805679.42634857
    <class 'sklearn.svm._classes.SVR'> 33061617.61326263
    <class 'sklearn.linear_model._base.LinearRegression'> 154154.59357750855
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 149802.74482220985
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 126639.19540273429
    <class 'xgboost.sklearn.XGBRegressor'> 145329.7937599686
    <class 'sklearn.svm._classes.SVR'> 91966.04990972951
    <class 'sklearn.linear_model._base.LinearRegression'> 459156.2348300455
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 338370.2351544244
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 377708.04353699466
    <class 'xgboost.sklearn.XGBRegressor'> 382308.3880232467
    <class 'sklearn.svm._classes.SVR'> 229585.93298941304


#### Chosen algorithm :  DecisionTreeRegressor (however it may depends on the choice of videos)

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
nt_confs = dict()
for i in range(len(index_test)):
    it = index_test[i]
    for ts in train_sizes:
        nt_confs[(i, ts)] = nt.predict_conf(target_id = it, train_size=ts,
                                      learning_algorithm = DecisionTreeRegressor)
```


```python
nt_confs
```




    {(0, 5): 5,
     (0, 10): 89,
     (0, 15): 68,
     (0, 20): 196,
     (0, 25): 104,
     (0, 30): 196,
     (1, 5): 41,
     (1, 10): 21,
     (1, 15): 93,
     (1, 20): 89,
     (1, 25): 21,
     (1, 30): 65,
     (2, 5): 8,
     (2, 10): 123,
     (2, 15): 8,
     (2, 20): 190,
     (2, 25): 12,
     (2, 30): 111,
     (3, 5): 7,
     (3, 10): 5,
     (3, 15): 38,
     (3, 20): 60,
     (3, 25): 4,
     (3, 30): 68,
     (4, 5): 39,
     (4, 10): 10,
     (4, 15): 16,
     (4, 20): 16,
     (4, 25): 42,
     (4, 30): 86,
     (5, 5): 20,
     (5, 10): 80,
     (5, 15): 74,
     (5, 20): 83,
     (5, 25): 42,
     (5, 30): 100,
     (6, 5): 165,
     (6, 10): 123,
     (6, 15): 109,
     (6, 20): 82,
     (6, 25): 82,
     (6, 30): 80,
     (7, 5): 1,
     (7, 10): 8,
     (7, 15): 62,
     (7, 20): 2,
     (7, 25): 79,
     (7, 30): 86,
     (8, 5): 1,
     (8, 10): 8,
     (8, 15): 177,
     (8, 20): 20,
     (8, 25): 25,
     (8, 30): 169,
     (9, 5): 41,
     (9, 10): 91,
     (9, 15): 85,
     (9, 20): 16,
     (9, 25): 189,
     (9, 30): 20,
     (10, 5): 13,
     (10, 10): 4,
     (10, 15): 109,
     (10, 20): 59,
     (10, 25): 96,
     (10, 30): 123,
     (11, 5): 1,
     (11, 10): 24,
     (11, 15): 8,
     (11, 20): 123,
     (11, 25): 1,
     (11, 30): 170,
     (12, 5): 1,
     (12, 10): 172,
     (12, 15): 1,
     (12, 20): 62,
     (12, 25): 101,
     (12, 30): 123,
     (13, 5): 4,
     (13, 10): 12,
     (13, 15): 100,
     (13, 20): 5,
     (13, 25): 26,
     (13, 30): 26,
     (14, 5): 32,
     (14, 10): 1,
     (14, 15): 123,
     (14, 20): 138,
     (14, 25): 89,
     (14, 30): 168,
     (15, 5): 1,
     (15, 10): 91,
     (15, 15): 62,
     (15, 20): 8,
     (15, 25): 172,
     (15, 30): 137,
     (16, 5): 1,
     (16, 10): 31,
     (16, 15): 1,
     (16, 20): 85,
     (16, 25): 83,
     (16, 30): 91,
     (17, 5): 6,
     (17, 10): 8,
     (17, 15): 97,
     (17, 20): 3,
     (17, 25): 123,
     (17, 30): 123,
     (18, 5): 3,
     (18, 10): 46,
     (18, 15): 104,
     (18, 20): 85,
     (18, 25): 29,
     (18, 30): 89,
     (19, 5): 4,
     (19, 10): 41,
     (19, 15): 102,
     (19, 20): 64,
     (19, 25): 14,
     (19, 30): 85,
     (20, 5): 3,
     (20, 10): 1,
     (20, 15): 50,
     (20, 20): 168,
     (20, 25): 123,
     (20, 30): 91,
     (21, 5): 1,
     (21, 10): 85,
     (21, 15): 1,
     (21, 20): 62,
     (21, 25): 91,
     (21, 30): 91,
     (22, 5): 1,
     (22, 10): 1,
     (22, 15): 67,
     (22, 20): 67,
     (22, 25): 8,
     (22, 30): 20,
     (23, 5): 65,
     (23, 10): 83,
     (23, 15): 65,
     (23, 20): 91,
     (23, 25): 62,
     (23, 30): 153,
     (24, 5): 1,
     (24, 10): 4,
     (24, 15): 63,
     (24, 20): 4,
     (24, 25): 169,
     (24, 30): 38,
     (25, 5): 3,
     (25, 10): 4,
     (25, 15): 38,
     (25, 20): 85,
     (25, 25): 60,
     (25, 30): 57,
     (26, 5): 49,
     (26, 10): 16,
     (26, 15): 2,
     (26, 20): 2,
     (26, 25): 112,
     (26, 30): 109,
     (27, 5): 4,
     (27, 10): 3,
     (27, 15): 104,
     (27, 20): 38,
     (27, 25): 1,
     (27, 30): 4,
     (28, 5): 41,
     (28, 10): 25,
     (28, 15): 1,
     (28, 20): 74,
     (28, 25): 165,
     (28, 30): 123,
     (29, 5): 62,
     (29, 10): 1,
     (29, 15): 1,
     (29, 20): 123,
     (29, 25): 91,
     (29, 30): 50,
     (30, 5): 59,
     (30, 10): 8,
     (30, 15): 89,
     (30, 20): 177,
     (30, 25): 165,
     (30, 30): 165,
     (31, 5): 11,
     (31, 10): 42,
     (31, 15): 2,
     (31, 20): 2,
     (31, 25): 16,
     (31, 30): 42,
     (32, 5): 2,
     (32, 10): 4,
     (32, 15): 4,
     (32, 20): 40,
     (32, 25): 5,
     (32, 30): 46,
     (33, 5): 1,
     (33, 10): 38,
     (33, 15): 38,
     (33, 20): 12,
     (33, 25): 38,
     (33, 30): 46,
     (34, 5): 34,
     (34, 10): 4,
     (34, 15): 92,
     (34, 20): 4,
     (34, 25): 104,
     (34, 30): 68,
     (35, 5): 12,
     (35, 10): 2,
     (35, 15): 47,
     (35, 20): 16,
     (35, 25): 2,
     (35, 30): 113,
     (36, 5): 3,
     (36, 10): 34,
     (36, 15): 2,
     (36, 20): 2,
     (36, 25): 2,
     (36, 30): 32,
     (37, 5): 1,
     (37, 10): 39,
     (37, 15): 2,
     (37, 20): 2,
     (37, 25): 2,
     (37, 30): 2,
     (38, 5): 16,
     (38, 10): 3,
     (38, 15): 2,
     (38, 20): 20,
     (38, 25): 92,
     (38, 30): 164,
     (39, 5): 100,
     (39, 10): 164,
     (39, 15): 123,
     (39, 20): 168,
     (39, 25): 170,
     (39, 30): 123,
     (40, 5): 9,
     (40, 10): 112,
     (40, 15): 32,
     (40, 20): 2,
     (40, 25): 2,
     (40, 30): 3,
     (41, 5): 2,
     (41, 10): 8,
     (41, 15): 2,
     (41, 20): 2,
     (41, 25): 2,
     (41, 30): 2,
     (42, 5): 83,
     (42, 10): 8,
     (42, 15): 20,
     (42, 20): 123,
     (42, 25): 134,
     (42, 30): 179,
     (43, 5): 4,
     (43, 10): 41,
     (43, 15): 42,
     (43, 20): 42,
     (43, 25): 123,
     (43, 30): 91,
     (44, 5): 37,
     (44, 10): 8,
     (44, 15): 3,
     (44, 20): 2,
     (44, 25): 2,
     (44, 30): 168,
     (45, 5): 1,
     (45, 10): 21,
     (45, 15): 2,
     (45, 20): 123,
     (45, 25): 177,
     (45, 30): 20,
     (46, 5): 4,
     (46, 10): 41,
     (46, 15): 179,
     (46, 20): 104,
     (46, 25): 46,
     (46, 30): 168,
     (47, 5): 4,
     (47, 10): 195,
     (47, 15): 155,
     (47, 20): 123,
     (47, 25): 123,
     (47, 30): 177,
     (48, 5): 8,
     (48, 10): 177,
     (48, 15): 2,
     (48, 20): 134,
     (48, 25): 134,
     (48, 30): 2,
     (49, 5): 5,
     (49, 10): 92,
     (49, 15): 0,
     (49, 20): 3,
     (49, 25): 60,
     (49, 30): 42,
     (50, 5): 8,
     (50, 10): 1,
     (50, 15): 91,
     (50, 20): 15,
     (50, 25): 8,
     (50, 30): 42,
     (51, 5): 0,
     (51, 10): 32,
     (51, 15): 34,
     (51, 20): 171,
     (51, 25): 5,
     (51, 30): 163,
     (52, 5): 166,
     (52, 10): 4,
     (52, 15): 131,
     (52, 20): 104,
     (52, 25): 171,
     (52, 30): 123,
     (53, 5): 4,
     (53, 10): 83,
     (53, 15): 91,
     (53, 20): 168,
     (53, 25): 123,
     (53, 30): 15,
     (54, 5): 2,
     (54, 10): 20,
     (54, 15): 16,
     (54, 20): 43,
     (54, 25): 70,
     (54, 30): 2,
     (55, 5): 0,
     (55, 10): 5,
     (55, 15): 4,
     (55, 20): 42,
     (55, 25): 104,
     (55, 30): 5,
     (56, 5): 4,
     (56, 10): 170,
     (56, 15): 8,
     (56, 20): 74,
     (56, 25): 123,
     (56, 30): 8,
     (57, 5): 62,
     (57, 10): 123,
     (57, 15): 85,
     (57, 20): 73,
     (57, 25): 107,
     (57, 30): 101,
     (58, 5): 9,
     (58, 10): 64,
     (58, 15): 21,
     (58, 20): 65,
     (58, 25): 89,
     (58, 30): 75,
     (59, 5): 41,
     (59, 10): 8,
     (59, 15): 119,
     (59, 20): 8,
     (59, 25): 34,
     (59, 30): 12,
     (60, 5): 123,
     (60, 10): 29,
     (60, 15): 20,
     (60, 20): 159,
     (60, 25): 9,
     (60, 30): 80,
     (61, 5): 0,
     (61, 10): 184,
     (61, 15): 5,
     (61, 20): 4,
     (61, 25): 5,
     (61, 30): 4,
     (62, 5): 1,
     (62, 10): 169,
     (62, 15): 41,
     (62, 20): 165,
     (62, 25): 165,
     (62, 30): 89,
     (63, 5): 4,
     (63, 10): 4,
     (63, 15): 5,
     (63, 20): 4,
     (63, 25): 34,
     (63, 30): 12,
     (64, 5): 4,
     (64, 10): 4,
     (64, 15): 68,
     (64, 20): 4,
     (64, 25): 68,
     (64, 30): 68,
     (65, 5): 20,
     (65, 10): 20,
     (65, 15): 20,
     (65, 20): 65,
     (65, 25): 91,
     (65, 30): 170,
     (66, 5): 32,
     (66, 10): 169,
     (66, 15): 131,
     (66, 20): 32,
     (66, 25): 104,
     (66, 30): 123,
     (67, 5): 2,
     (67, 10): 64,
     (67, 15): 3,
     (67, 20): 62,
     (67, 25): 2,
     (67, 30): 2,
     (68, 5): 83,
     (68, 10): 8,
     (68, 15): 168,
     (68, 20): 123,
     (68, 25): 21,
     (68, 30): 169,
     (69, 5): 3,
     (69, 10): 42,
     (69, 15): 123,
     (69, 20): 89,
     (69, 25): 2,
     (69, 30): 2,
     (70, 5): 1,
     (70, 10): 100,
     (70, 15): 85,
     (70, 20): 15,
     (70, 25): 123,
     (70, 30): 123,
     (71, 5): 4,
     (71, 10): 32,
     (71, 15): 4,
     (71, 20): 111,
     (71, 25): 187,
     (71, 30): 130,
     (72, 5): 3,
     (72, 10): 137,
     (72, 15): 20,
     (72, 20): 8,
     (72, 25): 165,
     (72, 30): 8,
     (73, 5): 1,
     (73, 10): 171,
     (73, 15): 9,
     (73, 20): 22,
     (73, 25): 38,
     (73, 30): 4,
     (74, 5): 4,
     (74, 10): 1,
     (74, 15): 91,
     (74, 20): 62,
     (74, 25): 93,
     (74, 30): 108,
     (75, 5): 3,
     (75, 10): 1,
     (75, 15): 164,
     (75, 20): 16,
     (75, 25): 16,
     (75, 30): 164,
     (76, 5): 64,
     (76, 10): 2,
     (76, 15): 196,
     (76, 20): 20,
     (76, 25): 20,
     (76, 30): 3,
     (77, 5): 4,
     (77, 10): 4,
     (77, 15): 4,
     (77, 20): 4,
     (77, 25): 12,
     (77, 30): 104,
     (78, 5): 32,
     (78, 10): 29,
     (78, 15): 54,
     (78, 20): 186,
     (78, 25): 46,
     (78, 30): 38,
     (79, 5): 41,
     (79, 10): 5,
     (79, 15): 5,
     (79, 20): 12,
     (79, 25): 4,
     (79, 30): 12,
     (80, 5): 12,
     (80, 10): 91,
     (80, 15): 29,
     (80, 20): 32,
     (80, 25): 169,
     (80, 30): 91,
     (81, 5): 1,
     (81, 10): 80,
     (81, 15): 21,
     (81, 20): 184,
     (81, 25): 169,
     (81, 30): 153,
     (82, 5): 14,
     (82, 10): 65,
     (82, 15): 3,
     (82, 20): 123,
     (82, 25): 91,
     (82, 30): 170,
     (83, 5): 165,
     (83, 10): 21,
     (83, 15): 2,
     (83, 20): 158,
     (83, 25): 123,
     (83, 30): 89,
     (84, 5): 8,
     (84, 10): 3,
     (84, 15): 8,
     (84, 20): 46,
     (84, 25): 4,
     (84, 30): 12,
     (85, 5): 123,
     (85, 10): 15,
     (85, 15): 62,
     (85, 20): 85,
     (85, 25): 26,
     (85, 30): 102,
     (86, 5): 4,
     (86, 10): 89,
     (86, 15): 25,
     (86, 20): 32,
     (86, 25): 32,
     (86, 30): 38,
     (87, 5): 12,
     (87, 10): 57,
     (87, 15): 5,
     (87, 20): 68,
     (87, 25): 171,
     (87, 30): 4,
     (88, 5): 1,
     (88, 10): 32,
     (88, 15): 40,
     (88, 20): 36,
     (88, 25): 38,
     (88, 30): 5,
     (89, 5): 41,
     (89, 10): 38,
     (89, 15): 38,
     (89, 20): 46,
     (89, 25): 38,
     (89, 30): 46,
     (90, 5): 8,
     (90, 10): 62,
     (90, 15): 165,
     (90, 20): 8,
     (90, 25): 15,
     (90, 30): 89,
     (91, 5): 74,
     (91, 10): 14,
     (91, 15): 93,
     (91, 20): 29,
     (91, 25): 168,
     (91, 30): 100,
     (92, 5): 1,
     (92, 10): 2,
     (92, 15): 39,
     (92, 20): 3,
     (92, 25): 10,
     (92, 30): 2,
     (93, 5): 1,
     (93, 10): 42,
     (93, 15): 42,
     (93, 20): 91,
     (93, 25): 62,
     (93, 30): 85,
     (94, 5): 64,
     (94, 10): 2,
     (94, 15): 2,
     (94, 20): 89,
     (94, 25): 2,
     (94, 30): 123,
     (95, 5): 1,
     (95, 10): 6,
     (95, 15): 123,
     (95, 20): 8,
     (95, 25): 123,
     (95, 30): 159,
     (96, 5): 1,
     (96, 10): 11,
     (96, 15): 2,
     (96, 20): 2,
     (96, 25): 70,
     (96, 30): 42,
     (97, 5): 20,
     (97, 10): 42,
     (97, 15): 2,
     (97, 20): 2,
     (97, 25): 2,
     (97, 30): 2,
     (98, 5): 3,
     (98, 10): 3,
     (98, 15): 165,
     (98, 20): 2,
     (98, 25): 169,
     (98, 30): 10,
     (99, 5): 168,
     (99, 10): 43,
     (99, 15): 8,
     (99, 20): 38,
     (99, 25): 8,
     (99, 30): 8,
     (100, 5): 20,
     (100, 10): 10,
     (100, 15): 3,
     (100, 20): 39,
     (100, 25): 16,
     (100, 30): 2,
     (101, 5): 46,
     (101, 10): 3,
     (101, 15): 3,
     (101, 20): 80,
     (101, 25): 123,
     (101, 30): 80,
     (102, 5): 1,
     (102, 10): 3,
     (102, 15): 177,
     (102, 20): 183,
     (102, 25): 168,
     (102, 30): 32,
     (103, 5): 1,
     (103, 10): 123,
     (103, 15): 141,
     (103, 20): 159,
     (103, 25): 62,
     (103, 30): 142,
     (104, 5): 12,
     (104, 10): 2,
     (104, 15): 62,
     (104, 20): 2,
     (104, 25): 159,
     (104, 30): 43,
     (105, 5): 2,
     (105, 10): 3,
     (105, 15): 177,
     (105, 20): 123,
     (105, 25): 165,
     (105, 30): 169,
     (106, 5): 14,
     (106, 10): 123,
     (106, 15): 2,
     (106, 20): 123,
     (106, 25): 62,
     (106, 30): 123,
     (107, 5): 84,
     (107, 10): 14,
     (107, 15): 2,
     (107, 20): 3,
     (107, 25): 23,
     (107, 30): 10,
     (108, 5): 1,
     (108, 10): 177,
     (108, 15): 8,
     (108, 20): 165,
     (108, 25): 159,
     (108, 30): 137,
     (109, 5): 11,
     (109, 10): 38,
     (109, 15): 46,
     (109, 20): 4,
     (109, 25): 38,
     (109, 30): 100,
     (110, 5): 1,
     (110, 10): 15,
     (110, 15): 89,
     (110, 20): 9,
     (110, 25): 123,
     (110, 30): 123,
     (111, 5): 1,
     (111, 10): 11,
     (111, 15): 47,
     (111, 20): 39,
     (111, 25): 164,
     (111, 30): 42,
     (112, 5): 2,
     (112, 10): 1,
     (112, 15): 62,
     (112, 20): 168,
     (112, 25): 62,
     (112, 30): 97,
     (113, 5): 8,
     (113, 10): 50,
     (113, 15): 15,
     (113, 20): 170,
     (113, 25): 123,
     (113, 30): 20,
     (114, 5): 62,
     (114, 10): 62,
     (114, 15): 89,
     (114, 20): 8,
     (114, 25): 89,
     (114, 30): 41,
     (115, 5): 1,
     (115, 10): 42,
     (115, 15): 8,
     (115, 20): 2,
     (115, 25): 2,
     (115, 30): 2,
     (116, 5): 4,
     (116, 10): 29,
     (116, 15): 60,
     (116, 20): 12,
     (116, 25): 130,
     (116, 30): 60,
     (117, 5): 51,
     (117, 10): 2,
     (117, 15): 10,
     (117, 20): 2,
     (117, 25): 23,
     (117, 30): 10,
     (118, 5): 172,
     (118, 10): 20,
     (118, 15): 9,
     (118, 20): 29,
     (118, 25): 123,
     (118, 30): 3,
     (119, 5): 1,
     (119, 10): 100,
     (119, 15): 8,
     (119, 20): 62,
     (119, 25): 165,
     (119, 30): 96,
     (120, 5): 12,
     (120, 10): 62,
     (120, 15): 20,
     (120, 20): 8,
     (120, 25): 89,
     (120, 30): 123,
     (121, 5): 1,
     (121, 10): 168,
     (121, 15): 2,
     (121, 20): 8,
     (121, 25): 2,
     (121, 30): 31,
     (122, 5): 1,
     (122, 10): 52,
     (122, 15): 16,
     (122, 20): 2,
     (122, 25): 123,
     (122, 30): 119,
     (123, 5): 4,
     (123, 10): 21,
     (123, 15): 181,
     (123, 20): 91,
     (123, 25): 29,
     (123, 30): 102,
     (124, 5): 119,
     (124, 10): 9,
     (124, 15): 8,
     (124, 20): 166,
     (124, 25): 89,
     (124, 30): 3,
     (125, 5): 1,
     (125, 10): 123,
     (125, 15): 123,
     (125, 20): 89,
     (125, 25): 168,
     (125, 30): 123,
     (126, 5): 8,
     (126, 10): 8,
     (126, 15): 1,
     (126, 20): 123,
     (126, 25): 62,
     (126, 30): 94,
     (127, 5): 34,
     (127, 10): 169,
     (127, 15): 123,
     (127, 20): 46,
     (127, 25): 104,
     (127, 30): 104,
     (128, 5): 32,
     (128, 10): 123,
     (128, 15): 123,
     (128, 20): 3,
     (128, 25): 170,
     (128, 30): 183,
     (129, 5): 1,
     (129, 10): 78,
     (129, 15): 3,
     (129, 20): 172,
     (129, 25): 123,
     (129, 30): 137,
     (130, 5): 1,
     (130, 10): 1,
     (130, 15): 165,
     (130, 20): 20,
     (130, 25): 91,
     (130, 30): 123,
     (131, 5): 1,
     (131, 10): 5,
     (131, 15): 4,
     (131, 20): 57,
     (131, 25): 8,
     (131, 30): 54,
     (132, 5): 177,
     (132, 10): 123,
     (132, 15): 123,
     (132, 20): 169,
     (132, 25): 123,
     (132, 30): 165,
     (133, 5): 1,
     (133, 10): 3,
     (133, 15): 79,
     (133, 20): 3,
     (133, 25): 3,
     (133, 30): 2,
     (134, 5): 113,
     (134, 10): 2,
     (134, 15): 2,
     (134, 20): 3,
     (134, 25): 70,
     (134, 30): 2,
     (135, 5): 3,
     (135, 10): 62,
     (135, 15): 93,
     (135, 20): 101,
     (135, 25): 62,
     (135, 30): 100,
     (136, 5): 15,
     (136, 10): 82,
     (136, 15): 15,
     (136, 20): 75,
     (136, 25): 20,
     (136, 30): 196,
     (137, 5): 4,
     (137, 10): 26,
     (137, 15): 2,
     (137, 20): 26,
     (137, 25): 172,
     (137, 30): 26,
     (138, 5): 104,
     (138, 10): 4,
     (138, 15): 1,
     (138, 20): 5,
     (138, 25): 32,
     (138, 30): 4,
     (139, 5): 46,
     (139, 10): 68,
     (139, 15): 46,
     (139, 20): 43,
     (139, 25): 4,
     (139, 30): 4,
     (140, 5): 0,
     (140, 10): 8,
     (140, 15): 46,
     (140, 20): 4,
     (140, 25): 32,
     (140, 30): 54,
     (141, 5): 29,
     (141, 10): 89,
     (141, 15): 2,
     (141, 20): 2,
     (141, 25): 10,
     (141, 30): 42,
     (142, 5): 1,
     (142, 10): 21,
     (142, 15): 1,
     (142, 20): 5,
     (142, 25): 5,
     (142, 30): 32,
     (143, 5): 8,
     (143, 10): 0,
     (143, 15): 153,
     (143, 20): 170,
     (143, 25): 89,
     (143, 30): 89,
     (144, 5): 3,
     (144, 10): 20,
     (144, 15): 130,
     (144, 20): 100,
     (144, 25): 62,
     (144, 30): 15,
     (145, 5): 179,
     (145, 10): 165,
     (145, 15): 165,
     (145, 20): 123,
     (145, 25): 123,
     (145, 30): 123,
     (146, 5): 1,
     (146, 10): 3,
     (146, 15): 91,
     (146, 20): 15,
     (146, 25): 21,
     (146, 30): 123,
     (147, 5): 1,
     (147, 10): 62,
     (147, 15): 84,
     (147, 20): 85,
     (147, 25): 100,
     (147, 30): 50,
     (148, 5): 100,
     (148, 10): 1,
     (148, 15): 91,
     (148, 20): 100,
     (148, 25): 91,
     (148, 30): 91,
     (149, 5): 23,
     (149, 10): 1,
     (149, 15): 123,
     (149, 20): 1,
     (149, 25): 123,
     (149, 30): 192,
     (150, 5): 67,
     (150, 10): 13,
     (150, 15): 50,
     (150, 20): 67,
     (150, 25): 100,
     (150, 30): 101,
     (151, 5): 6,
     (151, 10): 6,
     (151, 15): 2,
     (151, 20): 2,
     (151, 25): 2,
     (151, 30): 10,
     (152, 5): 2,
     (152, 10): 8,
     (152, 15): 1,
     (152, 20): 123,
     (152, 25): 89,
     (152, 30): 170,
     (153, 5): 2,
     (153, 10): 12,
     (153, 15): 5,
     (153, 20): 4,
     (153, 25): 4,
     (153, 30): 5,
     (154, 5): 1,
     (154, 10): 21,
     (154, 15): 169,
     (154, 20): 168,
     (154, 25): 26,
     (154, 30): 26,
     (155, 5): 2,
     (155, 10): 34,
     (155, 15): 161,
     (155, 20): 4,
     (155, 25): 32,
     (155, 30): 130,
     (156, 5): 1,
     (156, 10): 13,
     (156, 15): 1,
     (156, 20): 177,
     (156, 25): 2,
     (156, 30): 29,
     (157, 5): 23,
     (157, 10): 2,
     (157, 15): 2,
     (157, 20): 2,
     (157, 25): 2,
     (157, 30): 2,
     (158, 5): 62,
     (158, 10): 1,
     (158, 15): 177,
     (158, 20): 20,
     (158, 25): 177,
     (158, 30): 134,
     (159, 5): 1,
     (159, 10): 89,
     (159, 15): 143,
     (159, 20): 137,
     (159, 25): 49,
     (159, 30): 134,
     (160, 5): 1,
     (160, 10): 80,
     (160, 15): 6,
     (160, 20): 123,
     (160, 25): 2,
     (160, 30): 3,
     (161, 5): 24,
     (161, 10): 172,
     (161, 15): 62,
     (161, 20): 170,
     (161, 25): 8,
     (161, 30): 168,
     (162, 5): 6,
     (162, 10): 30,
     (162, 15): 1,
     (162, 20): 20,
     (162, 25): 1,
     (162, 30): 17,
     (163, 5): 105,
     (163, 10): 1,
     (163, 15): 57,
     (163, 20): 32,
     (163, 25): 36,
     (163, 30): 104,
     (164, 5): 2,
     (164, 10): 2,
     (164, 15): 8,
     (164, 20): 2,
     (164, 25): 173,
     (164, 30): 26,
     (165, 5): 29,
     (165, 10): 123,
     (165, 15): 83,
     (165, 20): 177,
     (165, 25): 21,
     (165, 30): 168,
     (166, 5): 8,
     (166, 10): 2,
     (166, 15): 164,
     (166, 20): 134,
     ...}




```python
nt_data = pd.DataFrame({"id_video" : [i for i in range(len(index_test))]})
for ts in train_sizes:
    nt_data["conf"+str(ts)] = [nt_confs[(i, ts)] for i in range(len(index_test))]
```


```python
nt_data.set_index("id_video").to_csv("../../../results/raw_data/NT_results.csv")
```


```python

```
