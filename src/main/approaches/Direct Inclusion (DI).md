### Direct Inclusion (DI)

**Direct Inclusion (DI)** includes input properties directly in the model during the training phase. The trained model then predicts the performance of x264 based on a set of properties (i.e. information about the input video) **and** a set of configuration options (i.e. information about the configuration). We fed this model with the 201 configurations of our dataset, and the properties of the test videos. We select the configuration giving the best prediction (e.g. the lowest bitrate).

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
v_names_test = np.loadtxt("../../../results/raw_data/test_names.csv", dtype= str)

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

to_dummy_features = ['rc_lookahead', 'analyse', 'me', 'subme', 'mixed_ref', 'me_range', 'qpmax', 
                    'aq-mode','trellis','fast_pskip', 'chroma_qp_offset', 'bframes', 'b_pyramid', 
                    'b_adapt', 'direct', 'ref', 'deblock', 'weightb', 'open_gop', 'weightp', 
                    'scenecut']

# the list of measurements
listVideo = []

# we add each dataset in the list, converting the time to the right format
# third line asserts that the measures are complete
for v in v_names:
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



#### Classify inputs in groups

#### Import properties


```python
# we load the file (in itself an aggregation of datasets)
# the file is available in the data folder, then ugc_meta
# each line is a video, and the columns are the different metrics
# provided by Wang et. al.
meta = pd.read_csv("../../../data/ugc/ugc_meta/all_features.csv").set_index('FILENAME')
# category is a high-level characterization of the content of the video
# for an example, Sports for a sports video
# you can see more details about different categories 
# and metrics per category in the resources/categories.csv file
# I also recommand to read the Youtube UGC paper to understand why we consider these categories
meta['category']=[str(meta.index[i]).split('_')[0] for i in range(meta.shape[0])]
# a lot of NA, not a big feature importance, seems complicated to compute -> remove NOISE DMOS
del meta['NOISE_DMOS']
# fill NA with zeros
meta = meta.fillna(0)
# create a numeric variable (quanti) to compute the category
# one video has one and only one category (1 to 1 in sql, so we can join the tables)
# again, to do it properly, we should use dummies
# but then we cannot compare directly the importances of the metrics to categories 
cat_tab = pd.Series(meta['category'].values).unique()
meta['video_category'] = [np.where(cat_tab==meta['category'][i])[0][0] for i in range(len(meta['category']))]
# delete the old columns (quali)
del meta['category']
# we normalize the variables, since height mean is about 1000, and complexity about 2
# different scales do not behave correctly with learning algorithms
for col in meta.columns:#[:len(meta.columns)-1]:
    inter = np.array(meta[col],float)
    meta[col] = (inter-np.mean(inter))/np.std(inter)
# print the results for the training inputs
meta
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
      <th>SLEEQ_DMOS</th>
      <th>BANDING_DMOS</th>
      <th>WIDTH</th>
      <th>HEIGHT</th>
      <th>SPATIAL_COMPLEXITY</th>
      <th>TEMPORAL_COMPLEXITY</th>
      <th>CHUNK_COMPLEXITY_VARIATION</th>
      <th>COLOR_COMPLEXITY</th>
      <th>video_category</th>
    </tr>
    <tr>
      <th>FILENAME</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Animation_1080P-01b3</th>
      <td>-0.678859</td>
      <td>4.653015</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>-1.475487</td>
      <td>-1.547345</td>
      <td>-0.892454</td>
      <td>-1.210798</td>
      <td>-1.618194</td>
    </tr>
    <tr>
      <th>Animation_1080P-05f8</th>
      <td>0.844509</td>
      <td>0.741729</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>-0.147257</td>
      <td>0.444086</td>
      <td>2.545710</td>
      <td>2.207516</td>
      <td>-1.618194</td>
    </tr>
    <tr>
      <th>Animation_1080P-0c4f</th>
      <td>-0.655778</td>
      <td>-0.377464</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>0.422320</td>
      <td>-0.963192</td>
      <td>1.054868</td>
      <td>-1.232460</td>
      <td>-1.618194</td>
    </tr>
    <tr>
      <th>Animation_1080P-0cdf</th>
      <td>-0.294170</td>
      <td>-0.059377</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>-0.028644</td>
      <td>0.430810</td>
      <td>-0.103261</td>
      <td>-0.448284</td>
      <td>-1.618194</td>
    </tr>
    <tr>
      <th>Animation_1080P-18f5</th>
      <td>-0.478821</td>
      <td>-0.377464</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>1.289017</td>
      <td>-0.958767</td>
      <td>-0.051295</td>
      <td>0.192920</td>
      <td>-1.618194</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Vlog_720P-561e</th>
      <td>-0.678859</td>
      <td>-0.377464</td>
      <td>-0.239786</td>
      <td>-0.333314</td>
      <td>0.978979</td>
      <td>-1.414583</td>
      <td>-0.652893</td>
      <td>0.457201</td>
      <td>1.494379</td>
    </tr>
    <tr>
      <th>Vlog_720P-5d08</th>
      <td>-0.678859</td>
      <td>-0.377464</td>
      <td>-0.773092</td>
      <td>-0.333314</td>
      <td>3.257287</td>
      <td>-0.303807</td>
      <td>-0.437698</td>
      <td>-0.158009</td>
      <td>1.494379</td>
    </tr>
    <tr>
      <th>Vlog_720P-60f8</th>
      <td>0.444433</td>
      <td>0.623920</td>
      <td>-0.239786</td>
      <td>-0.333314</td>
      <td>0.234418</td>
      <td>-0.042708</td>
      <td>-0.364385</td>
      <td>-0.149344</td>
      <td>1.494379</td>
    </tr>
    <tr>
      <th>Vlog_720P-6410</th>
      <td>-0.455739</td>
      <td>3.769441</td>
      <td>-0.239786</td>
      <td>-0.333314</td>
      <td>-0.770856</td>
      <td>2.121314</td>
      <td>1.971065</td>
      <td>-0.240326</td>
      <td>1.494379</td>
    </tr>
    <tr>
      <th>Vlog_720P-6d56</th>
      <td>0.629083</td>
      <td>-0.353902</td>
      <td>-0.239786</td>
      <td>-0.333314</td>
      <td>-0.329287</td>
      <td>0.329026</td>
      <td>1.646979</td>
      <td>0.565512</td>
      <td>1.494379</td>
    </tr>
  </tbody>
</table>
<p>1397 rows × 9 columns</p>
</div>



#### Learning Algorithm to predict the performance of input videos by including input properties in the model

data : input properties + configurations

predicted : performance (i.e. here the bitrate)

#### Generation of the dataset


##### Training set of inputs


```python
val_config = listVideo[0].drop(["kbs"],axis=1)

train_index = [v[:-4] for v in v_names_train]
test_index = [v[:-4] for v in v_names_test]

# we add the input properties
name_col = list(meta.columns)

# we add the x264 configuration options
for vcc in val_config.columns:
    name_col.append(vcc)

# Then, X (i.e the predicting variables) =  input properties + software configuration options
name_col.append("kbs")

# X length, the number of predicting variables
nb_col = len(name_col)
# the number of configurations
nb_config = 201

# generate the datasets = (X,y)
def gen_dataset(inputs_names):
    # inputs : names of videos
    # output : aggregation of multiple (X,y) for all the videos in the list of names provided in input 
    
    # the final dataset
    res = pd.DataFrame(np.zeros(nb_config*len(inputs_names)*nb_col).reshape(nb_config*len(inputs_names), nb_col))
    res.columns = name_col
    
    # we add the data video per video
    for i in range(len(inputs_names)):
        # first, we retrieve the name of the video
        video_name = inputs_names[i]
        index_video = np.where(np.array([v[:-4] for v in v_names], str)==video_name)[0][0]
        # we compute the performance, here
        bitrates = listVideo[index_video][predDimension]
        # get the input properties of the video
        video_prop = np.array(meta.loc[video_name], float)
        # compute the avrage value and the standard deviation for the bitrate
        # as we said in the paper, it does not change the order of variable
        # which is a good property
        moy = np.mean(bitrates)
        std = np.std(bitrates)
        # for each configuration, we add the values of the input properties and the configuration options (=X)
        # and the normalized values of bitrates (=y)
        for config_id in range(nb_config):
            val = list(tuple(video_prop) + tuple(val_config.loc[config_id]))
            val.append((bitrates[config_id]-moy)/std)
            res.loc[i*nb_config+config_id] = val
    return res

# training dataset
training_data = gen_dataset(train_index)

# dimensions of the different sets = a proxy to the measurement cost 
print("Training size : ", training_data.shape[0])

# OFFLINE - Training data
X = training_data[name_col[:len(name_col)-1]]
y = np.array(training_data["kbs"], float)
```

    Training size :  211050



```python
X
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
      <th>SLEEQ_DMOS</th>
      <th>BANDING_DMOS</th>
      <th>WIDTH</th>
      <th>HEIGHT</th>
      <th>SPATIAL_COMPLEXITY</th>
      <th>TEMPORAL_COMPLEXITY</th>
      <th>CHUNK_COMPLEXITY_VARIATION</th>
      <th>COLOR_COMPLEXITY</th>
      <th>video_category</th>
      <th>subme</th>
      <th>...</th>
      <th>direct_spatial</th>
      <th>deblock_0:0:0</th>
      <th>deblock_1:0:0</th>
      <th>weightb_1</th>
      <th>weightb_None</th>
      <th>open_gop_0</th>
      <th>open_gop_None</th>
      <th>scenecut_0</th>
      <th>scenecut_40</th>
      <th>scenecut_None</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.021466</td>
      <td>-0.236092</td>
      <td>-0.654363</td>
      <td>-0.777193</td>
      <td>-0.976374</td>
      <td>0.714036</td>
      <td>-0.698606</td>
      <td>-0.933520</td>
      <td>0.160419</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.021466</td>
      <td>-0.236092</td>
      <td>-0.654363</td>
      <td>-0.777193</td>
      <td>-0.976374</td>
      <td>0.714036</td>
      <td>-0.698606</td>
      <td>-0.933520</td>
      <td>0.160419</td>
      <td>6.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.021466</td>
      <td>-0.236092</td>
      <td>-0.654363</td>
      <td>-0.777193</td>
      <td>-0.976374</td>
      <td>0.714036</td>
      <td>-0.698606</td>
      <td>-0.933520</td>
      <td>0.160419</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.021466</td>
      <td>-0.236092</td>
      <td>-0.654363</td>
      <td>-0.777193</td>
      <td>-0.976374</td>
      <td>0.714036</td>
      <td>-0.698606</td>
      <td>-0.933520</td>
      <td>0.160419</td>
      <td>6.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.021466</td>
      <td>-0.236092</td>
      <td>-0.654363</td>
      <td>-0.777193</td>
      <td>-0.976374</td>
      <td>0.714036</td>
      <td>-0.698606</td>
      <td>-0.933520</td>
      <td>0.160419</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>211045</th>
      <td>-0.209538</td>
      <td>-0.377464</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>-0.824878</td>
      <td>-1.113656</td>
      <td>-0.857307</td>
      <td>0.751808</td>
      <td>-0.284234</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>211046</th>
      <td>-0.209538</td>
      <td>-0.377464</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>-0.824878</td>
      <td>-1.113656</td>
      <td>-0.857307</td>
      <td>0.751808</td>
      <td>-0.284234</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>211047</th>
      <td>-0.209538</td>
      <td>-0.377464</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>-0.824878</td>
      <td>-1.113656</td>
      <td>-0.857307</td>
      <td>0.751808</td>
      <td>-0.284234</td>
      <td>6.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>211048</th>
      <td>-0.209538</td>
      <td>-0.377464</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>-0.824878</td>
      <td>-1.113656</td>
      <td>-0.857307</td>
      <td>0.751808</td>
      <td>-0.284234</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>211049</th>
      <td>-0.209538</td>
      <td>-0.377464</td>
      <td>0.383054</td>
      <td>0.332504</td>
      <td>-0.824878</td>
      <td>-1.113656</td>
      <td>-0.857307</td>
      <td>0.751808</td>
      <td>-0.284234</td>
      <td>11.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>211050 rows × 54 columns</p>
</div>




```python
y
```




    array([ 2.51321814, -0.57034238, -0.58428136, ..., -0.36568327,
            1.29411201, -0.54569672])



#### Find the best Learning Algorithm


```python
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
model_nn.add(Dense(54, input_dim=54))
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
```

    Average MSE Linear Reg 0.0
    Average MSE Decision Tree 0.0
    Average MSE Random Forest 0.0
    Average MSE Boosting Tree 0.0
    Average MSE Support Vector Regressor 0.0
    Average MSE Neural Network 0.0



```python
print("Average MSE Linear Reg", mean_squared_error(y_test, ypred_lin))
print("Average MSE Decision Tree", mean_squared_error(y_test, ypred_dt))
print("Average MSE Random Forest", mean_squared_error(y_test, ypred_rf))
print("Average MSE Boosting Tree", mean_squared_error(y_test, y_pred_svr))
print("Average MSE Neural Network", mean_squared_error(y_test, y_pred_nn))
```

    Average MSE Linear Reg 0.4336576446857769
    Average MSE Decision Tree 0.21092497818642778
    Average MSE Random Forest 0.12935759406347513
    Average MSE Boosting Tree 0.38799765725386104
    Average MSE Neural Network 0.3586227204340556


#### Learning Algorithm kept : Random Forest
#### Hyperparameter optimization


```python
LA_rf = RandomForestRegressor()

grid_search_larf = GridSearchCV(estimator = LA_rf,
                                param_grid = {'n_estimators': [20, 50, 100],
                                              # we didn't include 1 for min_samples_leaf to avoid overfitting
                                         'min_samples_leaf' : [2, 5, 10],
                                         'max_depth' : [3, 5, None],
                                         'max_features' : [5, 15, 33]},
                                scoring = 'neg_mean_squared_error',
                                verbose = True,
                                n_jobs = 5)

grid_search_larf.fit(X, y)

print(grid_search_larf.best_params_)
```

    Fitting 5 folds for each of 81 candidates, totalling 405 fits


    [Parallel(n_jobs=5)]: Using backend LokyBackend with 5 concurrent workers.
    [Parallel(n_jobs=5)]: Done  40 tasks      | elapsed:   41.4s
    [Parallel(n_jobs=5)]: Done 190 tasks      | elapsed:  7.6min
    [Parallel(n_jobs=5)]: Done 405 out of 405 | elapsed: 33.3min finished


    {'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 100}


#### Results

{'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 100}

#### Predict the configurations for the test set


```python
LA_rf = RandomForestRegressor(max_depth = None,
                              max_features = 33, 
                              min_samples_leaf = 2,
                              n_estimators = 100)
LA_rf.fit(X, y)
```




    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=None, max_features=33, max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=2,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          n_estimators=100, n_jobs=None, oob_score=False,
                          random_state=None, verbose=0, warm_start=False)




```python
test_data = gen_dataset(test_index)
X_test_inputs = test_data[name_col[:len(name_col)-1]]
```


```python
assert X_test_inputs.shape[0] == 347*201
```

#### All right, we predict the performance, then we cut the results related to each input video, and we rank them


```python
res_perf_DI = LA_rf.predict(X_test_inputs)
```


```python
best_configs_DI = []

for i in range(len(v_names_test)):
    actual_video = [res_perf_DI[i*201+k] for k in range(201)]
    best_configs_DI.append(np.argmin(actual_video))
```


```python
assert len(best_configs_DI) == len(v_names_test)
```


```python
best_configs_DI
```




    [164,
     171,
     165,
     171,
     169,
     169,
     169,
     168,
     169,
     109,
     2,
     169,
     92,
     170,
     190,
     169,
     75,
     159,
     164,
     190,
     169,
     171,
     196,
     171,
     171,
     169,
     131,
     166,
     164,
     170,
     169,
     176,
     171,
     169,
     169,
     171,
     169,
     165,
     165,
     165,
     171,
     169,
     170,
     164,
     170,
     190,
     169,
     169,
     190,
     171,
     171,
     171,
     169,
     169,
     175,
     169,
     169,
     89,
     171,
     169,
     171,
     169,
     165,
     169,
     169,
     190,
     169,
     169,
     132,
     169,
     171,
     100,
     169,
     169,
     175,
     171,
     169,
     169,
     46,
     169,
     169,
     159,
     169,
     42,
     171,
     169,
     89,
     169,
     169,
     171,
     170,
     169,
     169,
     196,
     39,
     26,
     169,
     169,
     169,
     196,
     166,
     171,
     23,
     169,
     175,
     170,
     169,
     169,
     35,
     170,
     169,
     175,
     169,
     169,
     171,
     166,
     169,
     169,
     169,
     169,
     91,
     172,
     163,
     169,
     169,
     171,
     169,
     169,
     165,
     169,
     183,
     169,
     171,
     169,
     169,
     169,
     169,
     169,
     169,
     164,
     169,
     171,
     169,
     171,
     171,
     3,
     170,
     163,
     171,
     2,
     169,
     130,
     169,
     164,
     85,
     196,
     170,
     32,
     170,
     169,
     175,
     169,
     171,
     171,
     169,
     196,
     171,
     85,
     171,
     169,
     169,
     165,
     190,
     43,
     171,
     169,
     169,
     170,
     131,
     169,
     171,
     169,
     109,
     171,
     169,
     91,
     169,
     164,
     171,
     190,
     169,
     164,
     170,
     169,
     171,
     168,
     169,
     169,
     169,
     190,
     169,
     169,
     169,
     46,
     2,
     171,
     91,
     171,
     196,
     171,
     169,
     171,
     169,
     163,
     171,
     171,
     26,
     176,
     193,
     169,
     171,
     175,
     164,
     26,
     169,
     92,
     169,
     4,
     171,
     169,
     165,
     169,
     171,
     169,
     32,
     169,
     171,
     171,
     169,
     169,
     171,
     168,
     43,
     169,
     171,
     23,
     171,
     170,
     171,
     171,
     171,
     169,
     169,
     165,
     164,
     196,
     171,
     170,
     171,
     164,
     2,
     169,
     164,
     168,
     171,
     74,
     169,
     169,
     159,
     169,
     171,
     171,
     169,
     171,
     97,
     169,
     170,
     164,
     164,
     171,
     164,
     164,
     169,
     169,
     92,
     164,
     112,
     170,
     171,
     169,
     171,
     169,
     171,
     171,
     168,
     112,
     171,
     165,
     169,
     196,
     169,
     164,
     170,
     168,
     102,
     169,
     175,
     169,
     170,
     2,
     169,
     171,
     169,
     169,
     161,
     169,
     190,
     169,
     169,
     169,
     171,
     169,
     164,
     26,
     171,
     171,
     169,
     170,
     190,
     171,
     164,
     169,
     85,
     171,
     92,
     92,
     43,
     169,
     169,
     26,
     92,
     42,
     26,
     165,
     171,
     91,
     57]



#### Save the results


```python
#np.savetxt("../../../results/raw_data/DI_results2.csv", best_configs_DI, fmt = '%i')
```


```python

```
