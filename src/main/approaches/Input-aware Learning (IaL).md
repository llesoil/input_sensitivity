### Input-aware Learning (IaL)

**Input-aware Learning (IaL)** was first designed to overcome the input sensitivity of programs when compiling them with PetaBricks. 

> See the reference @inproceedings{ding2015,
  title={Autotuning algorithmic choice for input sensitivity},
  author={Ding, Yufei and Ansel, Jason and Veeramachaneni, Kalyan and Shen, Xipeng and O’Reilly, Una-May and Amarasinghe, Saman},
  booktitle={ACM SIGPLAN Notices},
  volume={50},
  number={6},
  pages={379--390},
  year={2015},
  organization={ACM},
  url={https://dl.acm.org/doi/pdf/10.1145/2813885.2737969},
}

Applied to the x264 case, it uses input properties of videos to propose a configuration working for a group of videos, sharing similar performances. 


According to Ding et al,  Input-Aware Learning can be broken down to six steps. 


Steps 1 to 4 are applied on the training set, while Step 5 and 6 consider a new input of the test set. 

**Step 1. Property extraction** - To mimic the domain knowledge of the expert, we use the videos' properties provided by the dataset of inputs

**Step 2. Form groups of inputs** - 
Based on the dendogram of Figure 1, we report on videos' properties that can be used to characterize four performance groups :
- Group 1. Action videos (high spatial and chunk complexities, Sports and News); 
- Group 2. Big resolution videos (low spatial and high temporal complexities, High Dynamic Range);
- Group 3. Still image videos (low temporal and chunk complexities, Lectures and HowTo)
- Group 4. Standard videos (average properties values, various contents)

Similarly, we used the training set of videos to build four groups of inputs. 

**Step 3. Landmark creation** - For each group, we artificially build a video, being the centroid of all the input videos of its group. We then use this video to select a set of landmarks (i.e. configurations), potential candidates to optimize the performance for this group. 

**Step 4. Performance measurements** - For each input video, we save the performances of its landmarks (i.e. the landmarks kept in Step 3, corresponding to its group).

**Step 5. Classify new inputs into a performance group** - Based on its input properties (see Step 1), we attribute a group to a new input video of the test set. It becomes a k-classification problem, k being the number of performance groups of Step 2. 

**Step 6. Propose a configuration for the new input** - We then propose a configuration based on the input properties of the video. It becomes a n-classification problem, where n is the number of landmarks kept for the group predicted in Step 5. We keep the best configuration predicted in Step 6.

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
from sklearn.svm import SVR, SVC
# decision trees
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
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
# left join performance groups to the dataset of metrics
perf = pd.read_csv("../../../results/raw_data/truth_group.csv").set_index('FILENAME')
meta_perf= perf.join(meta)
# print the results for the training inputs
meta_perf
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
      <th>perf_group</th>
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
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Animation_1080P-01b3</th>
      <td>2</td>
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
      <td>0</td>
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
      <td>3</td>
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
      <td>0</td>
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
      <td>0</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>Vlog_720P-561e</th>
      <td>3</td>
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
      <td>0</td>
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
      <td>0</td>
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
      <td>1</td>
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
      <td>3</td>
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
<p>1397 rows × 10 columns</p>
</div>



#### Learning Algorithm to classify videos in groups

data : input properties

predicted : group of the input


```python
if 'str_video_cat' in meta_perf.columns:
    del meta_perf['str_video_cat']

X = np.array(meta_perf[[k for k in meta_perf.columns if k !='perf_group']], float)
y = np.array(meta_perf['perf_group'], float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting trees
bt = GradientBoostingClassifier()
bt.fit(X_train, y_train)
y_pred_bt = bt.predict(X_test)

# Support Vector Classifier
svr = SVC()
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)


# neural network
model_nn = Sequential()
model_nn.add(Dense(9, input_dim=9))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(4, activation='softmax'))

model_nn.compile(loss='categorical_crossentropy', optimizer='Adam')
model_nn.fit(X_train, pd.get_dummies(y_train), epochs=5, verbose = False)
y_pred_nn = model_nn.predict(X_test)


conf = pd.crosstab(y_pred_dt, y_test)
val = np.sum(np.diag(conf))/len(y_test)
print('Test accuracy for Decision Tree: '+ str(val))

conf = pd.crosstab(y_pred_rf, y_test)
val = np.sum(np.diag(conf))/len(y_test)
print('Test accuracy for Random Forest: '+ str(val))

conf = pd.crosstab(y_pred_bt, y_test)
val = np.sum(np.diag(conf))/len(y_test)
print('Test accuracy for Boosting Trees: '+ str(val))

conf = pd.crosstab(y_pred_svr, y_test)
val = np.sum(np.diag(conf))/len(y_test)
print('Test accuracy for Support Vector: '+ str(val))

conf = pd.crosstab(np.argmax(y_pred_nn, axis = 1), y_test)
val = np.sum(np.diag(conf))/len(y_test)
print('Test accuracy for Neural Network: '+ str(val))

```

    Test accuracy for Decision Tree: 0.5428571428571428
    Test accuracy for Random Forest: 0.6547619047619048
    Test accuracy for Boosting Trees: 0.65
    Test accuracy for Support Vector: 0.6309523809523809
    Test accuracy for Neural Network: 0.5214285714285715


#### Learning Algorithm kept : Boosting Trees

#### Hyperparameter optimization

It is a compromise between the different input videos.


```python
short_name_train = [v[:-4] for v in v_names_train]
short_name_test = [v[:-4] for v in v_names_test]
```


```python
#### Training set/test set of configurations
X_train = np.array(meta_perf.loc[short_name_train][[k for k in meta_perf.columns if k !='perf_group']], float)
y_train = np.array(meta_perf.loc[short_name_train]['perf_group'], float)

LA_gb = GradientBoostingClassifier()

grid_search_larf = GridSearchCV(estimator = LA_gb,
                                param_grid = {'min_samples_split': [5, 10, 20],
                                              # we didn't include 1 for min_samples_leaf to avoid overfitting
                                         'min_samples_leaf' : [2, 5, 10],
                                         'max_depth' : [3, 5, None],
                                         'max_features' : [5, 15, 33]},
                                scoring = 'neg_mean_squared_error',
                                verbose = True,
                                n_jobs = 5)

#grid_search_larf.fit(X_train, y_train)

#print(grid_search_larf.best_params_)
```

#### Results

{'max_depth': 5, 'max_features': 5, 'min_samples_leaf': 10, 'min_samples_split': 10}


```python
X_test = np.array(meta_perf.loc[short_name_test][[k for k in meta_perf.columns if k !='perf_group']], float)

LA_gb = GradientBoostingClassifier(max_depth= 5,
                                   max_features = 5, 
                                   min_samples_leaf = 10,
                                   min_samples_split = 10)

LA_gb.fit(X_train, y_train)

pred_groups = LA_gb.predict(X_test)

conf = pd.crosstab(pred_groups, np.array(meta_perf.loc[short_name_test]['perf_group'], float))
val = np.sum(np.diag(conf))/len(y_test)
print("Accuracy =", str(val))
conf
```

    Accuracy = 0.7380952380952381





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
      <th>col_0</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>3.0</th>
    </tr>
    <tr>
      <th>row_0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>114</td>
      <td>0</td>
      <td>2</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0</td>
      <td>49</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>0</td>
      <td>4</td>
      <td>72</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



#### The accuracy between training and test is a quite good result


```python
#np.savetxt("../../../results/raw_data/predicted_test_group_IaL.csv", pred_groups, fmt='%i')
grps_test = np.loadtxt("../../../results/raw_data/predicted_test_group_IaL.csv", dtype = int)
grps_test
```




    array([1, 0, 2, 0, 3, 2, 0, 3, 3, 2, 1, 3, 1, 0, 0, 0, 3, 3, 2, 0, 2, 0,
           1, 0, 0, 1, 0, 2, 1, 3, 2, 3, 0, 3, 2, 0, 2, 2, 2, 3, 3, 2, 0, 1,
           3, 0, 2, 3, 0, 0, 0, 0, 0, 2, 1, 3, 0, 2, 0, 2, 0, 0, 0, 2, 2, 3,
           0, 2, 0, 3, 0, 0, 2, 3, 3, 0, 2, 2, 0, 2, 3, 3, 2, 2, 3, 0, 3, 3,
           0, 0, 0, 2, 0, 1, 1, 0, 3, 3, 3, 3, 3, 0, 1, 2, 1, 0, 3, 2, 0, 3,
           3, 1, 1, 3, 0, 0, 2, 2, 3, 2, 2, 3, 0, 2, 2, 0, 2, 2, 2, 3, 2, 2,
           0, 2, 0, 3, 1, 0, 2, 1, 3, 0, 2, 0, 0, 1, 3, 3, 0, 1, 2, 0, 3, 1,
           2, 3, 0, 0, 2, 3, 1, 3, 0, 0, 2, 1, 3, 1, 0, 0, 2, 2, 3, 2, 0, 3,
           1, 1, 0, 0, 0, 2, 2, 0, 3, 3, 1, 3, 0, 0, 2, 1, 0, 0, 0, 3, 3, 2,
           3, 0, 3, 3, 2, 0, 0, 3, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1, 3,
           0, 0, 1, 1, 0, 3, 3, 0, 0, 1, 1, 3, 0, 3, 0, 2, 0, 0, 3, 2, 0, 2,
           1, 3, 0, 1, 0, 0, 0, 0, 0, 3, 3, 2, 1, 0, 0, 3, 0, 1, 1, 0, 1, 2,
           3, 3, 1, 2, 3, 3, 3, 0, 0, 0, 3, 0, 0, 2, 2, 0, 1, 1, 1, 2, 1, 1,
           2, 3, 3, 2, 0, 2, 0, 0, 2, 2, 0, 2, 2, 2, 1, 1, 3, 2, 3, 3, 1, 0,
           3, 1, 3, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 1, 3, 0, 1, 3, 2, 0, 0,
           1, 2, 2, 3, 1, 2, 1, 0, 3, 3, 1, 1, 0, 1, 0, 1, 3])



#### Creation of the landmarks

For each group, we isolate the 5 configurations having the best performances.

First, we compute the ratios of the bitrate.


```python
bitrates = listVideo[0][predDimension]
ratio_bitrates = pd.DataFrame({"index" : range(201), 
                               "video0" : bitrates/min(bitrates)}).set_index("index")

for i in np.arange(1,len(listVideo),1):
    bitrates = listVideo[i][predDimension]/min(listVideo[i][predDimension])
    ratio_bitrates["video"+str(i)] = bitrates

ratio_bitrates
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
      <th>video0</th>
      <th>video1</th>
      <th>video2</th>
      <th>video3</th>
      <th>video4</th>
      <th>video5</th>
      <th>video6</th>
      <th>video7</th>
      <th>video8</th>
      <th>video9</th>
      <th>...</th>
      <th>video1040</th>
      <th>video1041</th>
      <th>video1042</th>
      <th>video1043</th>
      <th>video1044</th>
      <th>video1045</th>
      <th>video1046</th>
      <th>video1047</th>
      <th>video1048</th>
      <th>video1049</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>0</th>
      <td>1.862204</td>
      <td>3.586389</td>
      <td>2.567840</td>
      <td>4.862221</td>
      <td>3.169225</td>
      <td>3.189802</td>
      <td>3.412022</td>
      <td>2.880442</td>
      <td>3.824816</td>
      <td>3.072162</td>
      <td>...</td>
      <td>7.321984</td>
      <td>4.509879</td>
      <td>3.410863</td>
      <td>2.778554</td>
      <td>3.141600</td>
      <td>1.815321</td>
      <td>1.960430</td>
      <td>2.075508</td>
      <td>1.744034</td>
      <td>3.224831</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.130092</td>
      <td>1.933077</td>
      <td>1.389767</td>
      <td>1.172065</td>
      <td>1.417829</td>
      <td>1.302191</td>
      <td>1.373230</td>
      <td>1.170470</td>
      <td>1.676993</td>
      <td>1.538306</td>
      <td>...</td>
      <td>1.500755</td>
      <td>1.691921</td>
      <td>1.265955</td>
      <td>2.271980</td>
      <td>1.447346</td>
      <td>1.352919</td>
      <td>1.627961</td>
      <td>1.083952</td>
      <td>1.122840</td>
      <td>1.432093</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.126783</td>
      <td>1.136725</td>
      <td>1.544691</td>
      <td>1.420825</td>
      <td>1.846335</td>
      <td>1.434206</td>
      <td>1.284300</td>
      <td>1.577655</td>
      <td>1.843583</td>
      <td>1.071359</td>
      <td>...</td>
      <td>1.951180</td>
      <td>2.122874</td>
      <td>1.365082</td>
      <td>1.225935</td>
      <td>2.035878</td>
      <td>1.701721</td>
      <td>1.967654</td>
      <td>1.179261</td>
      <td>1.307660</td>
      <td>1.209609</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.135252</td>
      <td>1.255231</td>
      <td>1.549854</td>
      <td>1.417216</td>
      <td>1.860384</td>
      <td>1.441762</td>
      <td>1.300931</td>
      <td>1.593014</td>
      <td>1.849610</td>
      <td>1.115791</td>
      <td>...</td>
      <td>1.964765</td>
      <td>2.142246</td>
      <td>1.342774</td>
      <td>1.404374</td>
      <td>1.967706</td>
      <td>1.694929</td>
      <td>1.949904</td>
      <td>1.179663</td>
      <td>1.301094</td>
      <td>1.225856</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.193487</td>
      <td>2.546688</td>
      <td>1.141010</td>
      <td>1.820536</td>
      <td>1.130004</td>
      <td>1.842751</td>
      <td>1.839037</td>
      <td>1.194368</td>
      <td>1.432343</td>
      <td>1.721692</td>
      <td>...</td>
      <td>1.571653</td>
      <td>2.190037</td>
      <td>1.746266</td>
      <td>3.587797</td>
      <td>1.093759</td>
      <td>1.066703</td>
      <td>1.094103</td>
      <td>1.248850</td>
      <td>1.136048</td>
      <td>2.156670</td>
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
      <th>196</th>
      <td>1.042759</td>
      <td>1.312574</td>
      <td>1.277822</td>
      <td>1.098890</td>
      <td>1.339164</td>
      <td>1.148410</td>
      <td>1.291479</td>
      <td>1.034207</td>
      <td>1.295245</td>
      <td>1.301235</td>
      <td>...</td>
      <td>1.344430</td>
      <td>1.252478</td>
      <td>1.119108</td>
      <td>1.719691</td>
      <td>1.414472</td>
      <td>1.287799</td>
      <td>1.518524</td>
      <td>1.068775</td>
      <td>1.078492</td>
      <td>1.164970</td>
    </tr>
    <tr>
      <th>197</th>
      <td>1.063946</td>
      <td>1.332809</td>
      <td>1.300628</td>
      <td>1.149238</td>
      <td>1.343390</td>
      <td>1.241268</td>
      <td>1.303251</td>
      <td>1.084247</td>
      <td>1.452455</td>
      <td>1.368212</td>
      <td>...</td>
      <td>1.340898</td>
      <td>1.332315</td>
      <td>1.207371</td>
      <td>1.779311</td>
      <td>1.422829</td>
      <td>1.296252</td>
      <td>1.565146</td>
      <td>1.044772</td>
      <td>1.087553</td>
      <td>1.232681</td>
    </tr>
    <tr>
      <th>198</th>
      <td>1.133806</td>
      <td>1.963083</td>
      <td>1.430254</td>
      <td>1.149238</td>
      <td>1.421048</td>
      <td>1.271634</td>
      <td>1.411676</td>
      <td>1.203450</td>
      <td>1.561483</td>
      <td>1.597412</td>
      <td>...</td>
      <td>1.561765</td>
      <td>1.706384</td>
      <td>1.319690</td>
      <td>2.469687</td>
      <td>1.472009</td>
      <td>1.346861</td>
      <td>1.569729</td>
      <td>1.119159</td>
      <td>1.127905</td>
      <td>1.553307</td>
    </tr>
    <tr>
      <th>199</th>
      <td>1.531351</td>
      <td>2.759685</td>
      <td>1.372104</td>
      <td>2.067581</td>
      <td>1.297777</td>
      <td>2.056266</td>
      <td>2.186113</td>
      <td>1.349877</td>
      <td>1.663221</td>
      <td>2.066693</td>
      <td>...</td>
      <td>1.856370</td>
      <td>2.579451</td>
      <td>2.022114</td>
      <td>4.238554</td>
      <td>1.242817</td>
      <td>1.258331</td>
      <td>1.397348</td>
      <td>1.589516</td>
      <td>1.409675</td>
      <td>2.489680</td>
    </tr>
    <tr>
      <th>200</th>
      <td>1.010198</td>
      <td>1.670513</td>
      <td>1.398224</td>
      <td>1.121718</td>
      <td>1.360367</td>
      <td>1.135152</td>
      <td>1.329221</td>
      <td>1.062336</td>
      <td>1.637008</td>
      <td>1.474221</td>
      <td>...</td>
      <td>1.527168</td>
      <td>1.447871</td>
      <td>1.247527</td>
      <td>1.852915</td>
      <td>1.349037</td>
      <td>1.230864</td>
      <td>1.477360</td>
      <td>1.015365</td>
      <td>1.054236</td>
      <td>1.451752</td>
    </tr>
  </tbody>
</table>
<p>201 rows × 1050 columns</p>
</div>




```python
names_video_group = ["video"+str(i)
                      for i in range(len(v_names_train))
                      if y_train[i] == 2]
names_video_group
#[np.argmin(rankings[n]) for n in names_video_group]
```




    ['video3',
     'video5',
     'video6',
     'video9',
     'video10',
     'video15',
     'video27',
     'video35',
     'video39',
     'video41',
     'video53',
     'video54',
     'video56',
     'video59',
     'video66',
     'video93',
     'video95',
     'video104',
     'video105',
     'video107',
     'video112',
     'video121',
     'video123',
     'video125',
     'video129',
     'video140',
     'video142',
     'video143',
     'video148',
     'video159',
     'video161',
     'video170',
     'video172',
     'video180',
     'video185',
     'video189',
     'video191',
     'video197',
     'video198',
     'video202',
     'video203',
     'video206',
     'video208',
     'video213',
     'video216',
     'video219',
     'video223',
     'video228',
     'video230',
     'video231',
     'video234',
     'video252',
     'video255',
     'video266',
     'video268',
     'video272',
     'video274',
     'video276',
     'video280',
     'video281',
     'video288',
     'video296',
     'video297',
     'video299',
     'video305',
     'video313',
     'video317',
     'video327',
     'video328',
     'video332',
     'video355',
     'video358',
     'video366',
     'video370',
     'video375',
     'video379',
     'video380',
     'video384',
     'video389',
     'video406',
     'video407',
     'video408',
     'video410',
     'video412',
     'video416',
     'video418',
     'video424',
     'video425',
     'video432',
     'video433',
     'video434',
     'video439',
     'video449',
     'video458',
     'video461',
     'video462',
     'video463',
     'video467',
     'video472',
     'video475',
     'video478',
     'video485',
     'video486',
     'video491',
     'video493',
     'video495',
     'video498',
     'video508',
     'video513',
     'video514',
     'video516',
     'video529',
     'video532',
     'video533',
     'video535',
     'video540',
     'video541',
     'video542',
     'video550',
     'video555',
     'video556',
     'video563',
     'video566',
     'video594',
     'video595',
     'video600',
     'video607',
     'video610',
     'video617',
     'video624',
     'video626',
     'video634',
     'video642',
     'video643',
     'video644',
     'video647',
     'video649',
     'video661',
     'video670',
     'video678',
     'video680',
     'video684',
     'video686',
     'video689',
     'video692',
     'video703',
     'video706',
     'video707',
     'video708',
     'video709',
     'video727',
     'video736',
     'video737',
     'video739',
     'video747',
     'video749',
     'video752',
     'video753',
     'video754',
     'video756',
     'video764',
     'video765',
     'video766',
     'video791',
     'video807',
     'video808',
     'video813',
     'video815',
     'video820',
     'video829',
     'video834',
     'video838',
     'video839',
     'video841',
     'video842',
     'video843',
     'video850',
     'video852',
     'video853',
     'video860',
     'video862',
     'video865',
     'video871',
     'video877',
     'video879',
     'video891',
     'video895',
     'video898',
     'video903',
     'video913',
     'video928',
     'video935',
     'video941',
     'video953',
     'video954',
     'video956',
     'video958',
     'video963',
     'video965',
     'video966',
     'video969',
     'video973',
     'video977',
     'video989',
     'video998',
     'video1002',
     'video1004',
     'video1010',
     'video1030',
     'video1032',
     'video1034',
     'video1042',
     'video1049']




```python
def get_landmarks(id_group):
    
    names_video_group = ["video"+str(i) 
                      for i in range(len(v_names_train))
                      if y_train[i] == id_group]
    
    ratio_sum = [np.sum([ratio_bitrates[names_video_group].loc[i]]) for i in range(200)]
    
    ranks = [(i, ratio_sum[i]) for i in range(len(ratio_sum))]
    
    ranks.sort(key=lambda tup: tup[1], reverse=False)
    
    return [r[0] for r in ranks][0:5]

for i in range(4):
    print("Landmarks for the group", str(i), ":", get_landmarks(i))
```

    Landmarks for the group 0 : [171, 104, 60, 32, 161]
    Landmarks for the group 1 : [2, 164, 3, 175, 193]
    Landmarks for the group 2 : [169, 168, 170, 165, 123]
    Landmarks for the group 3 : [169, 170, 168, 165, 123]


It's interesting to see the different landmarks for the different groups : configuration 191 seems to be a good choice in general, but is not a landmark for group 1.


```python
landmarks = pd.DataFrame({'group0': get_landmarks(0),
              'group1': get_landmarks(1),
              'group2': get_landmarks(2),
              'group3': get_landmarks(3)})

landmarks.to_csv("../../../results/raw_data/predicted_landmarks_IaL.csv")
```


```python
landmarks
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
      <th>group0</th>
      <th>group1</th>
      <th>group2</th>
      <th>group3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>171</td>
      <td>2</td>
      <td>169</td>
      <td>169</td>
    </tr>
    <tr>
      <th>1</th>
      <td>104</td>
      <td>164</td>
      <td>168</td>
      <td>170</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>3</td>
      <td>170</td>
      <td>168</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>175</td>
      <td>165</td>
      <td>165</td>
    </tr>
    <tr>
      <th>4</th>
      <td>161</td>
      <td>193</td>
      <td>123</td>
      <td>123</td>
    </tr>
  </tbody>
</table>
</div>



#### Learning Algorithm per group


```python
def get_data(id_group):
    
    names_video_group_id = ["video"+str(i) 
                      for i in range(len(v_names_train))
                      if y_train[i] == id_group]
    
    names_video_group = [v_names_train[i][:-4]
                      for i in range(len(v_names_train))
                      if y_train[i] == id_group]
    
    X = meta.loc[names_video_group]
    
    l = landmarks['group'+str(id_group)]
    
    y = [l[np.argmin(rankings.loc[l][n])] for n in names_video_group_id]
    
    return (X, y)
```


```python
def get_data_test(id_group):
    
    names_video_group_id = [i
                      for i in range(len(v_names_test))
                      if grps_test[i] == id_group]
    
    names_video_group = [v_names_test[i][:-4]
                      for i in range(len(v_names_test))
                      if grps_test[i] == id_group]
    
    X = meta.loc[names_video_group]
    
    return (X, names_video_group_id)
```

Group 0


```python
X, y = get_data(0)

X_train, X_test, y_tr, y_te = train_test_split(X, y, test_size=0.3)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_tr)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_tr)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting trees
bt = GradientBoostingClassifier()
bt.fit(X_train, y_tr)
y_pred_bt = bt.predict(X_test)

# Support Vector Classifier
svr = SVC()
svr.fit(X_train, y_tr)
y_pred_svr = svr.predict(X_test)


# neural network
model_nn = Sequential()
model_nn.add(Dense(9, input_dim=9))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(5, activation='softmax'))

model_nn.compile(loss='categorical_crossentropy', optimizer='Adam')
model_nn.fit(X_train, pd.get_dummies(y_tr), epochs=5, verbose = False)
y_pred_nn = model_nn.predict(X_test)


conf = pd.crosstab(y_pred_dt, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Decision Tree: '+ str(val))

conf = pd.crosstab(y_pred_rf, y_te)
val = np.sum(np.diag(conf))/len(y_test)
print('Test accuracy for Random Forest: '+ str(val))

conf = pd.crosstab(y_pred_bt, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Boosting Trees: '+ str(val))

conf = pd.crosstab(y_pred_svr, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Support Vector: '+ str(val))

conf = pd.crosstab(np.argmax(y_pred_nn, axis = 1), y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Neural Network: '+ str(val))
```

    Test accuracy for Decision Tree: 0.0761904761904762
    Test accuracy for Random Forest: 0.009523809523809525
    Test accuracy for Boosting Trees: 0.02857142857142857
    Test accuracy for Support Vector: 0.06666666666666667
    Test accuracy for Neural Network: 0.24761904761904763


Neural network for the group 0


```python
X, y = get_data(0)

# neural network
model_nn = Sequential()
model_nn.add(Dense(9, input_dim=9))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(5, activation='softmax'))

model_nn.compile(loss='categorical_crossentropy', optimizer='Adam')
model_nn.fit(X, pd.get_dummies(y), epochs=5, verbose = False)
    
l = landmarks['group0']

res = get_data_test(0)

pred_grp0 = [l[pred] for pred in np.argmax(model_nn.predict(res[0]), axis = 1)]
indexgrp0 = res[1]
```

Group 1


```python
X, y = get_data(1)

X_train, X_test, y_tr, y_te = train_test_split(X, y, test_size=0.3)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_tr)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_tr)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting trees
bt = GradientBoostingClassifier()
bt.fit(X_train, y_tr)
y_pred_bt = bt.predict(X_test)

# Support Vector Classifier
svr = SVC()
svr.fit(X_train, y_tr)
y_pred_svr = svr.predict(X_test)


# neural network
model_nn = Sequential()
model_nn.add(Dense(9, input_dim=9))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(5, activation='softmax'))

model_nn.compile(loss='categorical_crossentropy', optimizer='Adam')
model_nn.fit(X_train, pd.get_dummies(y_tr), epochs=5, verbose = False)
y_pred_nn = model_nn.predict(X_test)


conf = pd.crosstab(y_pred_dt, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Decision Tree: '+ str(val))

conf = pd.crosstab(y_pred_rf, y_te)
val = np.sum(np.diag(conf))/len(y_test)
print('Test accuracy for Random Forest: '+ str(val))

conf = pd.crosstab(y_pred_bt, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Boosting Trees: '+ str(val))

conf = pd.crosstab(y_pred_svr, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Support Vector: '+ str(val))

conf = pd.crosstab(np.argmax(y_pred_nn, axis = 1), y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Neural Network: '+ str(val))

```

    Test accuracy for Decision Tree: 0.1
    Test accuracy for Random Forest: 0.014285714285714285
    Test accuracy for Boosting Trees: 0.16
    Test accuracy for Support Vector: 0.08
    Test accuracy for Neural Network: 0.04


Neural network for the group 1


```python
X, y = get_data(1)

# neural network
model_nn = Sequential()
model_nn.add(Dense(9, input_dim=9))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(5, activation='softmax'))

model_nn.compile(loss='categorical_crossentropy', optimizer='Adam')
model_nn.fit(X, pd.get_dummies(y), epochs=5, verbose = False)
    
l = landmarks['group1']

res = get_data_test(1)

pred_grp1 = [l[pred] for pred in np.argmax(model_nn.predict(res[0]), axis = 1)]
indexgrp1 = res[1]
```

Group 2


```python
X, y = get_data(2)

X_train, X_test, y_tr, y_te = train_test_split(X, y, test_size=0.3)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_tr)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_tr)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting trees
bt = GradientBoostingClassifier()
bt.fit(X_train, y_tr)
y_pred_bt = bt.predict(X_test)

# Support Vector Classifier
svr = SVC()
svr.fit(X_train, y_tr)
y_pred_svr = svr.predict(X_test)


# neural network
model_nn = Sequential()
model_nn.add(Dense(9, input_dim=9))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(5, activation='softmax'))

model_nn.compile(loss='categorical_crossentropy', optimizer='Adam')
model_nn.fit(X_train, pd.get_dummies(y_tr), epochs=5, verbose = False)
y_pred_nn = model_nn.predict(X_test)


conf = pd.crosstab(y_pred_dt, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Decision Tree: '+ str(val))

conf = pd.crosstab(y_pred_rf, y_te)
val = np.sum(np.diag(conf))/len(y_test)
print('Test accuracy for Random Forest: '+ str(val))

conf = pd.crosstab(y_pred_bt, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Boosting Trees: '+ str(val))

conf = pd.crosstab(y_pred_svr, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Support Vector: '+ str(val))

conf = pd.crosstab(np.argmax(y_pred_nn, axis = 1), y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Neural Network: '+ str(val))

```

    Test accuracy for Decision Tree: 0.34375
    Test accuracy for Random Forest: 0.08095238095238096
    Test accuracy for Boosting Trees: 0.4375
    Test accuracy for Support Vector: 0.984375
    Test accuracy for Neural Network: 0.046875


Neural network for the group 2


```python
X, y = get_data(2)

# neural network
model_nn = Sequential()
model_nn.add(Dense(9, input_dim=9))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(5, activation='softmax'))

model_nn.compile(loss='categorical_crossentropy', optimizer='Adam')
model_nn.fit(X, pd.get_dummies(y), epochs=5, verbose = False)
    
l = landmarks['group2']

res = get_data_test(2)

pred_grp2 = [l[pred] for pred in np.argmax(model_nn.predict(res[0]), axis = 1)]
indexgrp2 = res[1]
```

Group 3


```python
X, y = get_data(3)

X_train, X_test, y_tr, y_te = train_test_split(X, y, test_size=0.3)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_tr)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_tr)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting trees
bt = GradientBoostingClassifier()
bt.fit(X_train, y_tr)
y_pred_bt = bt.predict(X_test)

# Support Vector Classifier
svr = SVC()
svr.fit(X_train, y_tr)
y_pred_svr = svr.predict(X_test)


# neural network
model_nn = Sequential()
model_nn.add(Dense(9, input_dim=9))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(5, activation='softmax'))

model_nn.compile(loss='categorical_crossentropy', optimizer='Adam')
model_nn.fit(X_train, pd.get_dummies(y_tr), epochs=5, verbose = False)
y_pred_nn = model_nn.predict(X_test)


conf = pd.crosstab(y_pred_dt, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Decision Tree: '+ str(val))

conf = pd.crosstab(y_pred_rf, y_te)
val = np.sum(np.diag(conf))/len(y_test)
print('Test accuracy for Random Forest: '+ str(val))

conf = pd.crosstab(y_pred_bt, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Boosting Trees: '+ str(val))

conf = pd.crosstab(y_pred_svr, y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Support Vector: '+ str(val))

conf = pd.crosstab(np.argmax(y_pred_nn, axis = 1), y_te)
val = np.sum(np.diag(conf))/len(y_te)
print('Test accuracy for Neural Network: '+ str(val))

```

    Test accuracy for Decision Tree: 0.25510204081632654
    Test accuracy for Random Forest: 0.1
    Test accuracy for Boosting Trees: 0.32653061224489793
    Test accuracy for Support Vector: 0.826530612244898
    Test accuracy for Neural Network: 0.32653061224489793


Neural network for the third group


```python
X, y = get_data(3)

# neural network
model_nn = Sequential()
model_nn.add(Dense(9, input_dim=9))
# These 2 nodes are linked to the 10 nodes of the following layer
# The next nodes will receive a weighted sum of these 2 values as input
model_nn.add(Dense(10))
# we add an activation function to compose the result (i.e. the weighted sum) by reLU
# rectified Linear Unit = identity for positive and 0 for negative values
model_nn.add(Dense(5))
# Finally, we aggregate the 5 last values in one layer of one value, our prediction :)
model_nn.add(Dense(5, activation='softmax'))

model_nn.compile(loss='categorical_crossentropy', optimizer='Adam')
model_nn.fit(X, pd.get_dummies(y), epochs=5, verbose = False)
    
l = landmarks['group3']

res = get_data_test(3)

pred_grp3 = [l[pred] for pred in np.argmax(model_nn.predict(res[0]), axis = 1)]
indexgrp3 = res[1]
```

#### Save the results


```python
IaL_results = []

val0 = 0
val1 = 0
val2 = 0
val3 = 0

for i in range(len(v_names_test)):
    if i in indexgrp0:
        IaL_results.append(pred_grp0[val0])
        val0+=1
    if i in indexgrp1:
        IaL_results.append(pred_grp1[val1])
        val1+=1
    if i in indexgrp2:
        IaL_results.append(pred_grp2[val2])
        val2+=1
    if i in indexgrp3:
        IaL_results.append(pred_grp3[val3])
        val3+=1
```


```python
assert len(IaL_results) == len(v_names_test)
```


```python
print(IaL_results)
```

    [3, 60, 123, 32, 165, 169, 161, 165, 165, 169, 164, 165, 3, 32, 171, 171, 170, 170, 169, 60, 168, 32, 164, 60, 161, 3, 32, 170, 193, 165, 170, 169, 32, 165, 168, 32, 165, 123, 123, 170, 169, 168, 161, 164, 165, 32, 168, 165, 60, 161, 60, 161, 32, 169, 3, 165, 60, 165, 32, 169, 32, 161, 32, 168, 168, 170, 161, 169, 171, 123, 32, 32, 169, 123, 123, 161, 169, 169, 32, 168, 165, 123, 165, 169, 169, 32, 123, 169, 32, 161, 32, 169, 104, 175, 193, 171, 165, 170, 165, 170, 165, 161, 3, 123, 175, 161, 165, 169, 60, 165, 170, 3, 164, 123, 104, 32, 123, 165, 123, 168, 168, 170, 104, 170, 168, 161, 169, 169, 169, 123, 170, 123, 32, 168, 161, 165, 3, 104, 169, 3, 170, 161, 168, 32, 104, 3, 165, 123, 161, 3, 170, 161, 123, 175, 169, 170, 161, 161, 170, 170, 2, 170, 60, 32, 169, 175, 169, 164, 32, 32, 170, 165, 170, 168, 161, 165, 193, 164, 161, 104, 32, 168, 169, 60, 170, 165, 2, 169, 161, 161, 169, 164, 161, 161, 161, 165, 165, 169, 169, 32, 123, 170, 169, 32, 171, 170, 32, 161, 169, 161, 161, 104, 123, 32, 161, 32, 161, 32, 3, 170, 161, 32, 164, 3, 161, 169, 165, 161, 60, 193, 3, 123, 32, 165, 161, 123, 32, 104, 123, 169, 161, 169, 175, 169, 161, 2, 32, 161, 161, 161, 161, 123, 169, 169, 3, 171, 60, 165, 161, 193, 175, 32, 3, 168, 169, 170, 3, 168, 165, 169, 165, 171, 161, 161, 170, 60, 60, 169, 169, 161, 3, 175, 2, 169, 3, 2, 169, 170, 170, 168, 32, 169, 161, 32, 170, 169, 32, 123, 169, 170, 175, 3, 165, 168, 165, 123, 164, 60, 165, 164, 165, 171, 168, 60, 60, 170, 32, 170, 170, 161, 161, 161, 3, 170, 161, 164, 165, 169, 60, 104, 3, 165, 169, 170, 3, 169, 2, 32, 123, 165, 3, 3, 32, 3, 104, 164, 123]



```python
#np.savetxt("../../../results/raw_data/IaL_results.csv", IaL_results, fmt = '%i')
```


```python

```


```python

```
