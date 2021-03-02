# Predicting Performances of Configurable Systems:the Issue of Input Sensitivity

### This notebook details the main results presented in the paper submitted to ESEC/FSE.

#### Warning; Before launching the notebook, make sure you have installed all the packages in your python environment
#### To do that,  open a terminal in the replication folder, and use the requirements.txt file to download the libraries needed for this script :
`pip3 install -r requirements.txt`
#### If it worked, you should be able to launch the following cell to import libraries.


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
from sklearn.model_selection import train_test_split
# Simple clustering (iterative steps)
from sklearn.cluster import KMeans
# Support vector machine - support vector regressor
from sklearn.svm import SVR

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


#### Now, we import data


```python
#because x264 output is "m:s", where m is the number of minutes and s the number of seconds 
# we define a function to convert this format into the number of seconds
def elapsedtime_to_sec(el):
    tab = el.split(":")
    return float(tab[0])*60+float(tab[1])

# the data folder, see the markdown there for additional explanations
res_dir = "../../data/ugc/res_ugc/"

# the list of videos names, e.g. Animation_360P-3e40
# we sort the list so we keep the same ids between two launches
v_names = sorted(os.listdir(res_dir)) 

# the list of measurements
listVideo = []

# we add each dataset in the list, converting the time to the right format
# third line asserts that the measures are complete
for v in v_names:
    data = pd.read_table(res_dir+v, delimiter = ',')
    data['etime'] = [*map(elapsedtime_to_sec, data['elapsedtime'])]
    assert data.shape == (201,34), v
    listVideo.append(data)
```


```python
print(" We consider ", len(listVideo), " videos")
```

     We consider  1397  videos


#### Just a few words about the time needed to compute the data; about a month is needed to fully replicate the experience


```python
totalTime = np.sum([np.sum(vid["etime"]) for vid in listVideo])
print("Hours : "+str(totalTime/(3600)))
print("Days : "+str(totalTime/(24*3600)))
```

    Hours : 810.7560972222223
    Days : 33.781504050925925


#### Our focus in this paper is the bitrate, in kilobits per second


```python
#our variable of interest
predDimension = "fps"

for i in range(len(listVideo)):
    sizes = listVideo[i][predDimension]
    ind = sorted(range(len(sizes)), key=lambda k: sizes[k])
    listVideo[i]['ranking'] = ind
```


```python
v_names[1167]
```




    'VerticalVideo_1080P-2195.csv'



# In the paper, here starts Section II

# RQ1 - Do Input Videos Change Performances of x264 Configurations?

## RQ1.1 - Do software performances stay consistent across inputs?

#### A-] For this research question, we computed a matrix of Spearman correlations between each pair of input videos.

Other alternatives : Kullback-Leibler divergences to detect outliers, and Pearson correlation to compute only linear correlations 


```python
# number of videos
nbVideos = len(listVideo)
# matrix of coorelations
corrSpearman = [[0 for x in range(nbVideos)] for y in range(nbVideos)]

for i in range(nbVideos):
    for j in range(nbVideos):
        # A distribution of bitrates will have a correlaiton of 1 with itself
        if (i == j):
            corrSpearman[i][j] = 1
        else:
            # we compute the Spearman correlation between the input video i and the input video j
            corrSpearman[i][j] = sc.spearmanr(listVideo[i][predDimension],
                                            listVideo[j][predDimension]).correlation
```

#### Here is the distribution depicted in figure 1, on the bottom left; we removed the diagonal $i!=j$ in the following code.


```python
corrDescription = [corrSpearman[i][j] for i in range(nbVideos) for j in range(nbVideos) if i >j]
pd.Series(corrDescription).describe()
```




    count    975106.000000
    mean          0.933335
    std           0.082897
    min           0.005051
    25%           0.927942
    50%           0.958211
    75%           0.974024
    max           0.996437
    dtype: float64



#### Few statistics about input videos, mentioned in the text

#### A small detail; in the paper, when we mention the video having the id $i$,  it means the $(i+1)^{th}$ video of the list, because the first input has the index 0


```python
min_val = 1
ind_i = 0
ind_j = 0
for i in range(len(corrSpearman)):
    for j in range(len(corrSpearman[0])):
        if corrSpearman[i][j] < min_val:
            min_val = corrSpearman[i][j]
            ind_i = i
            ind_j = j

print("Value : ", min_val)
print("i : ", ind_i, ", j : ", ind_j)
```

    Value :  0.0050506020947232114
    i :  339 , j :  689



```python
corrSpearman[378][1192]
```




    0.904353004676807




```python
corrSpearman[314][1192]
```




    0.8459158382256203




```python
corrSpearman[378][314]
```




    0.8608662306497918



#### "For 95% of the videos, it is always possible to find another video having a correlation higher than 0.92" -> here is the proof 


```python
argm = [np.max([k for k in corrSpearman[i] if k <1]) for i in range(len(corrSpearman))]
pd.Series(argm).describe()
```




    count    1397.000000
    mean        0.980582
    std         0.035919
    min         0.541805
    25%         0.985185
    50%         0.990412
    75%         0.992357
    max         0.996437
    dtype: float64




```python
np.percentile(argm, 5)
```




    0.9367966381198882



## Figure 1

#### Now, let's compute figure 1!


```python
# the results directory
result_dir = "../../results/"

# We define a function to plot the correlogram
def plot_correlationmatrix_dendogram(corr, img_name, ticks, method= 'ward'):
    # inputs : a correlation matrix, or a matrix with quantitative values
    # a name for the image
    # the ticks to plot on axis
    # the aggregation method
    # output : a plot of an ordered correlogram with dendograms
    
    # we transform our matrix into a dataframe
    df = pd.DataFrame(corr)
    
    # group the videos, we choose the ward method 
    # single link method (minimum of distance) leads to numerous tiny clusters
    # centroid or average tend to split homogeneous clusters
    # and complete link aggregates unbalanced groups. 
    links = linkage(df, method=method,)
    order = leaves_list(links)
    
    # we order the correlation following the aggregation clustering
    mask = np.zeros_like(corr, dtype=np.bool)
    
    for i in range(nbVideos):
        for j in range(nbVideos):
            # Generate a mask for the upper triangle
            if i>j:
                mask[order[i]][order[j]] = True
    
    # seaborn clustermap plots a nice graph combining the correlogram and dendograms
    # cmap is the colormap, mask hides the lower triangular, method is the aggregation method,
    # linewidth is set to 0 because otherwise we can't see squre colors
    # figsize is the size of figure
    # we cannot print 1400 ticks, wo we hide them
    # to not fool the reviewers, we set the minimum to -1, to plot the full range of correlation
    # -0.69 would give a false impression of high input sensitivity
    g = sns.clustermap(df, cmap="vlag", mask=mask, method=method,
                   linewidths=0, figsize=(13, 13), cbar_kws={"ticks":ticks}, vmin =-1)
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.tick_params(right=False, bottom=False)
    # abcissa and ordered labels
    g.ax_heatmap.set_xlabel("Input videos", fontsize = 18)
    g.ax_heatmap.set_ylabel("Input videos", fontsize = 18)
    # we save the figure in the result folder
    plt.savefig(result_dir+img_name)
    # we show the graph
    plt.show()
    
    # finally we cut the dendogram to have 4 groups, and return thietr indexes
    return cut_tree(links, n_clusters = 4)

group_no_ordered = plot_correlationmatrix_dendogram(corrSpearman, 
                                 "corrmatrix-ugc-dendo-Spearman-" + predDimension + ".pdf",
                                 [k/5 for k in np.arange(-10,10,1)], method='ward')
```


    
![png](fps_files/fps_28_0.png)
    


#### To match the increasing number of groups to the order of the figure (from the left to the right), we change the ids of groups


```python
map_group = [2, 0, 3, 1]

def f(gr):
    return map_group[int(gr)]

# we apply this mapping
groups = np.array([*map(f, group_no_ordered)],int)

print("Group 1 contains", sum(groups==0), "input videos.")
print("Group 2 contains", sum(groups==1), "input videos.")
print("Group 3 contains", sum(groups==2), "input videos.")
print("Group 4 contains", sum(groups==3), "input videos.")
```

    Group 1 contains 1256 input videos.
    Group 2 contains 14 input videos.
    Group 3 contains 108 input videos.
    Group 4 contains 19 input videos.



```python
[v_names[i] for i in np.where(groups==3)[0]]
```




    ['HDR_1080P-1be2.csv',
     'HDR_1080P-3181.csv',
     'HDR_1080P-3a4a.csv',
     'HDR_1080P-46a4.csv',
     'HDR_1080P-49d6.csv',
     'HDR_1080P-55c4.csv',
     'HDR_2160P-06ae.csv',
     'HDR_2160P-1743.csv',
     'HDR_2160P-2a72.csv',
     'HDR_2160P-3663.csv',
     'HDR_2160P-3bf1.csv',
     'HDR_2160P-4581.csv',
     'HDR_2160P-51ea.csv',
     'HDR_2160P-5e25.csv',
     'HDR_2160P-6c6e.csv',
     'HDR_2160P-6eeb.csv',
     'HDR_2160P-6fa4.csv',
     'VR_2160P-4656.csv',
     'VerticalVideo_480P-790a.csv']



### B-] We also study rankings of configurations

#### First, we compute the rankings


```python
# first example ; we compute the rankings of the bitrate distribution for the first input video
bitrates = listVideo[0][predDimension]
# sorted rankings for the bitrates distribution (0: minimal, 200 : maximal)
ind = sorted(range(len(bitrates)), key=lambda k: bitrates[k])
# df
rankings = pd.DataFrame({"index" : range(201), "video0" : ind}).set_index("index")

for i in np.arange(1,len(listVideo),1):
    bitrates = listVideo[i][predDimension]
    ind = sorted(range(len(bitrates)), key=lambda k: bitrates[k])
    rankings["video"+str(i)] = ind

rankings.head()
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
      <th>video1387</th>
      <th>video1388</th>
      <th>video1389</th>
      <th>video1390</th>
      <th>video1391</th>
      <th>video1392</th>
      <th>video1393</th>
      <th>video1394</th>
      <th>video1395</th>
      <th>video1396</th>
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
      <td>108</td>
      <td>103</td>
      <td>103</td>
      <td>103</td>
      <td>103</td>
      <td>103</td>
      <td>103</td>
      <td>103</td>
      <td>103</td>
      <td>98</td>
      <td>...</td>
      <td>103</td>
      <td>103</td>
      <td>103</td>
      <td>104</td>
      <td>103</td>
      <td>103</td>
      <td>103</td>
      <td>103</td>
      <td>100</td>
      <td>103</td>
    </tr>
    <tr>
      <th>1</th>
      <td>107</td>
      <td>105</td>
      <td>105</td>
      <td>100</td>
      <td>104</td>
      <td>98</td>
      <td>108</td>
      <td>98</td>
      <td>98</td>
      <td>103</td>
      <td>...</td>
      <td>108</td>
      <td>48</td>
      <td>200</td>
      <td>103</td>
      <td>104</td>
      <td>104</td>
      <td>98</td>
      <td>104</td>
      <td>104</td>
      <td>104</td>
    </tr>
    <tr>
      <th>2</th>
      <td>102</td>
      <td>200</td>
      <td>101</td>
      <td>104</td>
      <td>108</td>
      <td>109</td>
      <td>200</td>
      <td>100</td>
      <td>101</td>
      <td>41</td>
      <td>...</td>
      <td>109</td>
      <td>104</td>
      <td>108</td>
      <td>102</td>
      <td>98</td>
      <td>200</td>
      <td>200</td>
      <td>200</td>
      <td>103</td>
      <td>101</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103</td>
      <td>100</td>
      <td>104</td>
      <td>102</td>
      <td>200</td>
      <td>108</td>
      <td>102</td>
      <td>104</td>
      <td>200</td>
      <td>95</td>
      <td>...</td>
      <td>102</td>
      <td>98</td>
      <td>98</td>
      <td>108</td>
      <td>200</td>
      <td>108</td>
      <td>101</td>
      <td>105</td>
      <td>105</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>101</td>
      <td>100</td>
      <td>109</td>
      <td>102</td>
      <td>104</td>
      <td>100</td>
      <td>102</td>
      <td>100</td>
      <td>101</td>
      <td>...</td>
      <td>105</td>
      <td>105</td>
      <td>109</td>
      <td>100</td>
      <td>108</td>
      <td>109</td>
      <td>105</td>
      <td>102</td>
      <td>108</td>
      <td>105</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1397 columns</p>
</div>



#### To get the most "unstable" ranking, we take the configuration having the highest standard deviation.


```python
# standard deviations for rankings of the 201 configurations
stds = [np.std(rankings.loc[i,:]) for i in range(len(rankings.index))]
print("Config min std : ", np.argmin(stds))
print("Config max std : ", np.argmax(stds))
print("Config med std : ", np.median(stds))

# depicts the most stable configuration ranking across inputs
plt.scatter(range(len(rankings.columns)), rankings.loc[np.argmin(stds), :])
plt.ylabel("Rank")
plt.xlabel("Video id")
plt.title("Configuration "+str(np.argmin(stds))+" : std = "+str(int(np.round(np.min(stds)))))
plt.savefig("../../results/config_min_std_ranking_"+predDimension+".png")
plt.show()

# depicts the most input sensitive configuration
plt.scatter(range(len(rankings.columns)), rankings.loc[np.argmax(stds), :])
plt.ylabel("Rank")
plt.xlabel("Video id")
plt.title("Configuration "+str(np.argmax(stds))+" : std = "+str(int(np.round(np.max(stds)))))
plt.savefig("../../results/config_max_std_ranking_"+predDimension+".png")
plt.show()
```

    Config min std :  0
    Config max std :  95
    Config med std :  43.94527433476486



    
![png](fps_files/fps_36_1.png)
    



    
![png](fps_files/fps_36_2.png)
    


#### Some statistics (not mentioned in the text)


```python
print("For config. 200, ", int(np.sum([1  for j in range(len(rankings.loc[np.argmin(stds),:])) 
              if rankings.loc[np.argmin(stds),:][j] > 105 and rankings.loc[np.argmin(stds),:][j] < 130])
      /len(rankings.loc[np.argmin(stds),:])*100),"% of configurations are between 105 and 130!")
```

    For config. 200,  2 % of configurations are between 105 and 130!



```python
np.where(rankings.loc[np.argmin(stds),:] == np.min(rankings.loc[np.argmin(stds),:]))
```




    (array([409]),)




```python
np.max(rankings.loc[np.argmax(stds),:])
```




    199




```python
np.where(rankings.loc[np.argmax(stds),:] == np.min(rankings.loc[np.argmax(stds),:]))
```




    (array([  48,  159,  160, 1145, 1312, 1335]),)




```python
np.max(rankings.loc[np.argmax(stds),:])
```




    199




```python
np.where(rankings.loc[np.argmin(stds),:] == np.max(rankings.loc[np.argmin(stds),:]))
```




    (array([  44,   51,   57,  130,  133,  137,  233,  371,  417,  528,  556,
             666,  749,  754,  756,  758,  760, 1219, 1243, 1385]),)



#### Rankings distributions


```python
pd.Series(rankings.loc[np.argmax(stds),:]).describe()
```




    count    1397.000000
    mean       84.887616
    std        75.978363
    min         1.000000
    25%        13.000000
    50%        54.000000
    75%       171.000000
    max       199.000000
    Name: 95, dtype: float64




```python
pd.Series(rankings.loc[np.argmin(stds),:]).describe()
```




    count    1397.000000
    mean      102.473157
    std        15.815878
    min        25.000000
    25%       103.000000
    50%       103.000000
    75%       103.000000
    max       200.000000
    Name: 0, dtype: float64



## RQ1-2- Are there some configuration options more sensitive to input videos?

#### A-] For RQ1-2, we compute the feature importances of configuration options for each video


```python
# the list of p = 24 features
listFeatures = ["cabac", "ref", "deblock", "analyse", "me", "subme", "mixed_ref", "me_range", "trellis", 
                "8x8dct", "fast_pskip", "chroma_qp_offset", "bframes", "b_pyramid", "b_adapt", "direct", 
                "weightb", "open_gop", "weightp", "scenecut", "rc_lookahead", "mbtree", "qpmax", "aq-mode"]

# we added the bitrates to predict it
to_keep = [k for k in listFeatures]
to_keep.append(predDimension)

# Those feature have values, so we had to transform them into quantitative variables
# They do not change a lot (in terms of feature importances and feature effects)
# But if they did, we would have transformed them into a set of dummies
# see https://www.xlstat.com/en/solutions/features/complete-disjuncive-tables-creating-dummy-variables
categorial = ['analyse', 'me', 'direct']

# A function that computes features importances 
# relative to a random forest learning the bitrate of a video compression
# (reused for each group)
def compute_Importances(listVid, id_short = None):
    # input : a list of videos
    # output : a dataframe of feature importances
    
    # we can give a list of ids, to work on a subset of the videos (e.g. a group)
    if not id_short:
        id_short = np.arange(0, len(listVid), 1)
    
    # the final list of importances
    listImportances = []

    # for each video of the list
    for id_video in range(len(listVid)):
        
        # we replace missing numbers by 0
        df = listVid[id_video][to_keep].replace(to_replace = "None", value = '0')
        # two values for deblock, "1:0:0" and "0:0:0", 
        # we just take the first character as a int
        df['deblock'] = [int(val[0]) for val in df['deblock']]
        
        for col in df.columns:
            # we center and reduce the quantitative variables
            # i.e substract the mean, and divide by the standard deviation
            # to avoid the scale of the vars to interfere with the learning process
            if col not in categorial:
                arr_col = np.array(df[col], int)
                arr_col = (arr_col-np.mean(arr_col))/(np.std(arr_col)+1e-5)
                df[col] = arr_col
            else:
                # if the variable is categorial
                if col not in [predDimension, 'ranking']:
                    df[col] = [np.where(k==df[col].unique())[0][0] for k in df[col]]
                    arr_col = np.array(df[col], int)
                    arr_col = (arr_col-np.mean(arr_col))/(np.std(arr_col)+1e-5)
                    df[col] = arr_col
        
        # for each video, we train a model
        clf = RandomForestRegressor(n_estimators=200)
        # we use all the configuration options as predicting variables
        X = df.drop([predDimension], axis=1)
        # and the bitrate distribution of the video as a variable to predict
        y = df[predDimension]
        # we train the model on all the data (not train-test since we don't use the model)
        clf.fit(X, y)
        
        # we add feature importances to the list
        listImportances.append(clf.feature_importances_)
    # final dataframe of feature importances
    res = pd.DataFrame({'features' : listFeatures})
    
    # significative numbers p, cs = 10^p
    cs = 100
    
    # we add the feature imps to the dataframe 
    for id_video in range(len(listImportances)):
        res['video_'+str(id_short[id_video])] = np.round(cs*listImportances[id_video])/cs
    
    # transpose it
    res = res.set_index('features').transpose()#.drop(['open_gop','qpmax'],axis=1)
    
    # return it 
    return res
```


```python
# we compute the feature importances
res_imp = compute_Importances(listVideo)
```

## Figure 2a
#### Then, we depict a boxplot of features importances; for each feature, there are 1397 feature importances (one per video)


```python
# we sort the features by names 
# here the first columns is not used, but it is useful 
# if we want to put the important features on top of the graph
listImp = [(np.abs(np.percentile(res_imp[col],75)-np.percentile(res_imp[col],25)),res_imp[col], col) 
           for col in res_imp.columns]
listImp.sort(key=lambda tup: tup[2], reverse=True)

# the names of the features
names = [l[2] for l in listImp]

# pretty names, we replace the names
to_replace_b4 = ["b_adapt", "b_pyramid", "chroma_qp_offset", "fast_pskip", 
                 "me_range", "mixed_ref", "open_gop", "rc_lookahead"]
to_replace_after = ["badapt", "bpyramid", "crqpoffset", "fastpskip", 
                    "merange", "mixedref", "opengop", "rclookahead"]

for n in range(len(to_replace_b4)):
    names[np.where(np.array(names, str) == to_replace_b4[n])[0][0]] = to_replace_after[n]

# fancy boxplot
red_square = dict(markerfacecolor='r', marker='s')
plt.figure(figsize=(15,8))
plt.grid()
plt.scatter([np.mean(l[1]) for l in listImp], range(1, 1+len(listImp)),
           marker="x", color = "black", alpha = 1, s = 20)
plt.boxplot([l[1] for l in listImp], flierprops=red_square, 
          vert=False, patch_artist=True, #widths=0.25,
          boxprops=dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
          whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))
# we add a line separating the influential features (median on the right of the orange line)
# from the rest of the features
plt.vlines(x = float(1/24), ymin = 0.5, ymax = 24, color="orange", linewidth=3)
plt.text(s = "Influential", x = 0.05, y = 8.1, size = 20, color = 'orange')
plt.xlabel("Random Forest feature importances", size=20)
plt.yticks(range(1, len(listImp) + 1), names, size= 16)
plt.savefig("../../results/boxplot_features_imp_rf_"+predDimension+".png")
plt.show()
```


    
![png](fps_files/fps_52_0.png)
    


#### B-] Since feature importances do not get how the predicting variables (i.e. the configuraiton options) affect the variable to predict (i.e. the bitrate), we add linear regression coefficients


```python
# alternatively, we can only select the important features to plot the tukey diagrams
short_features = ["mbtree", "aq-mode", "subme"]

# another function to compute the linear regression coefficients (reused for each group)
def compute_poly(listVid, id_short=None):
    # input : a list of videos, (list of ids)
    # output : a dataframe of feature importances
    
    # see compute_importances function, same principle
    if not id_short:
        id_short = np.arange(0,len(listVid),1)
    
    # see compute_importances function, same principle
    listImportances = []
    
    to_keep = [k for k in listFeatures]
    to_keep.append(predDimension)
    
    # this part of the code is not used there, 
    # but allows to add features interactions in the model
    names = listFeatures
    final_names = []
    final_names.append('constant')
    for n in names:
        final_names.append(n)
    for n1 in range(len(names)):
        for n2 in range(len(names)):
            if n1>=n2:
                final_names.append(str(names[n1])+'*'+str(names[n2]))
    
    for id_video in range(len(listVid)):
        # see compute_importances function, same principle
        df = listVid[id_video][to_keep].replace(to_replace ="None",value='0')
        df['deblock'] =[int(val[0]) for val in df['deblock']]
        # see compute_importances function, same principle
        for col in df.columns:
            if col not in categorial:
                arr_col = np.array(df[col],int)
                arr_col = (arr_col-np.mean(arr_col))/(np.std(arr_col)+1e-5)
                df[col] = arr_col
            else:
                df[col] = [np.where(k==df[col].unique())[0][0] for k in df[col]]
                arr_col = np.array(df[col],int)
                arr_col = (arr_col-np.mean(arr_col))/(np.std(arr_col)+1e-5)
                df[col] = arr_col
        # This time, we train an ordinary least square regression
        # i.e. we fit weights to predicting variables so it minimizes
        # the mean squared error between the prediction and the measures of bitrates 
        clf = LinearRegression()
        X = df.drop([predDimension],axis=1)
        #X = df[short_features]

        y = df[predDimension]
        # this part of the code is not used there, 
        # but allows to add features interactions in the model
        #poly = PolynomialFeatures(degree=1, interaction_only = False, include_bias = True)    
        #X_interact = pd.DataFrame(poly.fit_transform(X))#, columns=final_names)
        #kept_names = ['subme','aq-mode','mbtree','cabac','cabac*mbtree','aq-mode*subme','cabac*subme']
        
        # we train the model
        clf.fit(X,y)
        listImportances.append(clf.coef_)

    #res = pd.DataFrame({'features' : short_features})
    res = pd.DataFrame({'features' : listFeatures})

    # see compute_importances function, same principle
    cs = 100
    for id_video in range(len(listImportances)):
        res['video_'+str(id_short[id_video])] = np.round(cs*listImportances[id_video])/cs
    
    # see compute_importances function, same principle
    res = res.set_index('features').transpose()#.drop(['open_gop','qpmax'])
    #res = res.set_index('features').drop(['open_gop','qpmax']).transpose()
    return res

# we compute the coefficients
res_coef = compute_poly(listVideo)
# we can save the coefficients, useful for an analysis input per input
#res_coef.to_csv("../../results/list_features_importances_poly_"+predDimension+".csv")
# and we print them
res_coef
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
      <th>features</th>
      <th>cabac</th>
      <th>ref</th>
      <th>deblock</th>
      <th>analyse</th>
      <th>me</th>
      <th>subme</th>
      <th>mixed_ref</th>
      <th>me_range</th>
      <th>trellis</th>
      <th>8x8dct</th>
      <th>...</th>
      <th>b_adapt</th>
      <th>direct</th>
      <th>weightb</th>
      <th>open_gop</th>
      <th>weightp</th>
      <th>scenecut</th>
      <th>rc_lookahead</th>
      <th>mbtree</th>
      <th>qpmax</th>
      <th>aq-mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>video_0</th>
      <td>-0.01</td>
      <td>-0.10</td>
      <td>-0.01</td>
      <td>-0.03</td>
      <td>-0.09</td>
      <td>-0.23</td>
      <td>0.09</td>
      <td>-0.09</td>
      <td>-0.15</td>
      <td>-0.03</td>
      <td>...</td>
      <td>-0.23</td>
      <td>0.06</td>
      <td>0.21</td>
      <td>-0.0</td>
      <td>-0.00</td>
      <td>-0.02</td>
      <td>0.03</td>
      <td>-0.18</td>
      <td>0.0</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>video_1</th>
      <td>0.02</td>
      <td>-0.14</td>
      <td>-0.00</td>
      <td>-0.06</td>
      <td>-0.09</td>
      <td>-0.48</td>
      <td>-0.03</td>
      <td>0.01</td>
      <td>-0.24</td>
      <td>-0.09</td>
      <td>...</td>
      <td>-0.01</td>
      <td>-0.02</td>
      <td>0.01</td>
      <td>-0.0</td>
      <td>-0.00</td>
      <td>-0.03</td>
      <td>0.03</td>
      <td>-0.07</td>
      <td>0.0</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>video_2</th>
      <td>0.03</td>
      <td>-0.16</td>
      <td>-0.02</td>
      <td>-0.05</td>
      <td>-0.10</td>
      <td>-0.42</td>
      <td>-0.01</td>
      <td>-0.01</td>
      <td>-0.34</td>
      <td>-0.06</td>
      <td>...</td>
      <td>0.01</td>
      <td>-0.01</td>
      <td>0.10</td>
      <td>-0.0</td>
      <td>-0.01</td>
      <td>-0.06</td>
      <td>-0.00</td>
      <td>-0.12</td>
      <td>0.0</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>video_3</th>
      <td>0.01</td>
      <td>-0.11</td>
      <td>-0.01</td>
      <td>-0.06</td>
      <td>-0.08</td>
      <td>-0.50</td>
      <td>-0.02</td>
      <td>0.00</td>
      <td>-0.22</td>
      <td>-0.10</td>
      <td>...</td>
      <td>-0.01</td>
      <td>-0.02</td>
      <td>-0.05</td>
      <td>-0.0</td>
      <td>-0.03</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>-0.13</td>
      <td>0.0</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>video_4</th>
      <td>0.04</td>
      <td>-0.14</td>
      <td>-0.01</td>
      <td>-0.04</td>
      <td>-0.07</td>
      <td>-0.43</td>
      <td>-0.02</td>
      <td>-0.02</td>
      <td>-0.33</td>
      <td>-0.07</td>
      <td>...</td>
      <td>-0.00</td>
      <td>-0.01</td>
      <td>0.04</td>
      <td>-0.0</td>
      <td>-0.00</td>
      <td>-0.03</td>
      <td>-0.00</td>
      <td>-0.10</td>
      <td>0.0</td>
      <td>-0.03</td>
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
      <th>video_1392</th>
      <td>0.07</td>
      <td>-0.19</td>
      <td>-0.01</td>
      <td>-0.03</td>
      <td>-0.07</td>
      <td>-0.41</td>
      <td>0.02</td>
      <td>-0.03</td>
      <td>-0.37</td>
      <td>-0.04</td>
      <td>...</td>
      <td>-0.03</td>
      <td>0.02</td>
      <td>0.12</td>
      <td>-0.0</td>
      <td>0.02</td>
      <td>-0.04</td>
      <td>0.00</td>
      <td>-0.06</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>video_1393</th>
      <td>0.01</td>
      <td>-0.10</td>
      <td>0.01</td>
      <td>-0.02</td>
      <td>-0.08</td>
      <td>-0.45</td>
      <td>-0.10</td>
      <td>0.05</td>
      <td>-0.32</td>
      <td>-0.14</td>
      <td>...</td>
      <td>0.13</td>
      <td>-0.01</td>
      <td>-0.10</td>
      <td>-0.0</td>
      <td>-0.01</td>
      <td>-0.02</td>
      <td>-0.00</td>
      <td>-0.14</td>
      <td>0.0</td>
      <td>-0.03</td>
    </tr>
    <tr>
      <th>video_1394</th>
      <td>0.03</td>
      <td>-0.11</td>
      <td>-0.01</td>
      <td>-0.05</td>
      <td>-0.07</td>
      <td>-0.50</td>
      <td>-0.05</td>
      <td>0.04</td>
      <td>-0.26</td>
      <td>-0.09</td>
      <td>...</td>
      <td>0.01</td>
      <td>-0.03</td>
      <td>-0.05</td>
      <td>-0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>-0.00</td>
      <td>-0.09</td>
      <td>0.0</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>video_1395</th>
      <td>-0.02</td>
      <td>-0.08</td>
      <td>0.00</td>
      <td>-0.07</td>
      <td>-0.09</td>
      <td>-0.49</td>
      <td>-0.06</td>
      <td>0.04</td>
      <td>-0.14</td>
      <td>-0.14</td>
      <td>...</td>
      <td>0.20</td>
      <td>-0.04</td>
      <td>-0.20</td>
      <td>-0.0</td>
      <td>-0.02</td>
      <td>-0.18</td>
      <td>-0.00</td>
      <td>-0.09</td>
      <td>0.0</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>video_1396</th>
      <td>-0.01</td>
      <td>-0.10</td>
      <td>-0.00</td>
      <td>-0.06</td>
      <td>-0.08</td>
      <td>-0.54</td>
      <td>-0.05</td>
      <td>0.02</td>
      <td>-0.22</td>
      <td>-0.11</td>
      <td>...</td>
      <td>0.03</td>
      <td>-0.02</td>
      <td>-0.05</td>
      <td>-0.0</td>
      <td>0.01</td>
      <td>-0.04</td>
      <td>0.01</td>
      <td>-0.06</td>
      <td>0.0</td>
      <td>-0.04</td>
    </tr>
  </tbody>
</table>
<p>1397 rows × 24 columns</p>
</div>



## Figure 2b
#### Same idea for this plot, see the last cell of RQ1.2-A-]


```python
listImp = [(np.abs(np.percentile(res_coef[col],75)-np.percentile(res_coef[col],25)),res_coef[col], col) 
           for col in res_coef.columns]
listImp.sort(key=lambda tup: tup[2], reverse=True)

names = [l[2] for l in listImp]

to_replace_b4 = ["b_adapt", "b_pyramid", "chroma_qp_offset", "fast_pskip", 
                 "me_range", "mixed_ref", "open_gop", "rc_lookahead"]
to_replace_after = ["badapt", "bpyramid", "crqpoffset", "fastpskip", 
                    "merange", "mixedref", "opengop", "rclookahead"]

for n in range(len(to_replace_b4)):
    names[np.where(np.array(names, str) == to_replace_b4[n])[0][0]] = to_replace_after[n]
    
red_square = dict(markerfacecolor='r', marker='s')
plt.figure(figsize=(15,8))
plt.grid()
plt.scatter([np.mean(l[1]) for l in listImp], range(1, 1+len(listImp)),
           marker="x", color = "black", alpha = 1, s = 20)
plt.boxplot([l[1] for l in listImp], flierprops=red_square, 
          vert=False, patch_artist=True, #widths=0.25,
          boxprops=dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
          whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))
# A feature will have a positive impact on the bitrate associated to a video if his coefficient is positive
# A feature will have a negative impact on the bitrate associated to a video if his coefficient is negative
# The border or limit between these two ranges is the vertical line at x = 0
plt.vlines(x = 0, ymin = 0.5, ymax = 24.5, color="orange", linewidth= 3)
plt.text(s = "Negative impact", x = -0.96, y = 1.6, size = 20, color = 'orange')
plt.text(s = "Positive impact", x = 0.55, y = 1.6, size = 20, color = 'orange')
plt.xlabel("Linear Regression coefficients", size = 20)
plt.yticks(range(1, len(listImp) + 1), names, size= 16)
plt.xlim(-1.1, 1.1)
plt.savefig("../../results/boxplot_features_imp_linear_"+predDimension+".png")
plt.show()
```


    
![png](fps_files/fps_56_0.png)
    


# In the paper, here starts Section III

# RQ2. What could be a good tradeoff between the accuracy and the cost of the prediction? 

## Separation of training set of videos and test set of videos


```python
# v_names_train, v_names_test = train_test_split(v_names, train_size = 1050)

# v_names_train contains the inputs' names of the training set 
# the training set is used to learn the differences between inputs, i.e. it replaces domain knowledge
# v_names_test -> same for the test set
# the test set is used to evaluate the different state-of-the-art approaches

# save names of train inputs
# np.savetxt("../../data/train_names.csv", v_names_train, fmt='%s')
v_names_train = np.loadtxt("../../results/raw_data/train_names.csv", dtype= str)

# save names of test inputs
# np.savetxt("../../data/test_names.csv", v_names_test, fmt='%s')
v_names_test = np.loadtxt("../../results/raw_data/test_names.csv", dtype= str)

```


```python
v_names_train[0:50]
```




    array(['MusicVideo_480P-4cc8.csv', 'CoverSong_720P-6ae4.csv',
           'Sports_2160P-3d85.csv', 'HowTo_360P-18e7.csv',
           'TelevisionClip_360P-11d5.csv', 'TelevisionClip_360P-29f1.csv',
           'HowTo_720P-37d0.csv', 'HowTo_480P-7579.csv', 'VR_2160P-674b.csv',
           'Sports_480P-6508.csv', 'Gaming_2160P-28de.csv',
           'Gaming_1080P-55ac.csv', 'NewsClip_720P-2182.csv',
           'HDR_2160P-70ca.csv', 'HowTo_1080P-63ec.csv',
           'CoverSong_720P-7360.csv', 'Animation_1080P-6a33.csv',
           'HowTo_720P-479b.csv', 'Gaming_1080P-0ef8.csv',
           'Gaming_480P-5a5a.csv', 'Lecture_480P-2655.csv',
           'Sports_720P-6365.csv', 'Vlog_1080P-4a91.csv',
           'Vlog_1080P-4921.csv', 'HowTo_480P-3435.csv',
           'VerticalVideo_1080P-3a9b.csv', 'Gaming_360P-3eb6.csv',
           'HowTo_360P-7dcd.csv', 'CoverSong_720P-3dca.csv',
           'VR_1080P-5667.csv', 'NewsClip_720P-0c81.csv',
           'TelevisionClip_1080P-4c24.csv', 'VR_2160P-613e.csv',
           'Sports_480P-1019.csv', 'VR_2160P-097e.csv',
           'LyricVideo_360P-5b54.csv', 'Vlog_1080P-52fe.csv',
           'VerticalVideo_720P-44de.csv', 'Animation_720P-06a6.csv',
           'Sports_1080P-0640.csv', 'VR_2160P-7eab.csv',
           'TelevisionClip_480P-280f.csv', 'Gaming_1080P-3d58.csv',
           'NewsClip_720P-672c.csv', 'Gaming_1080P-6db2.csv',
           'HowTo_720P-7782.csv', 'Sports_360P-1803.csv', 'VR_1080P-0519.csv',
           'LyricVideo_1080P-28e8.csv', 'Gaming_360P-63e6.csv'], dtype='<U29')




```python
listVideoTest = [listVideo[i] for i in range(len(listVideo)) if v_names[i] in v_names_test]
assert len(listVideoTest) == len(v_names_test)
```

#### Isolate best configurations


```python
best_perfs = [np.min(vid[predDimension]) for vid in listVideoTest]
best_configs = [np.argmin(vid[predDimension]) for vid in listVideoTest]
```

## State-of-the-art approaches

### *SImple Learning*


```python
res_dir = "../../results/raw_data/"+predDimension+"/"
```

### Model Reuse (MR)


We arbitrarily choose a first video, learn a performance model on it, and select the best configuration minimizing the performance for this video. This approach represents the error made by a model trained on a  source input (i.e., a  first video) and transposed to a target input (i.e., a second video, different from the first one), without considering the difference of content between the source and the target. In theory, it corresponds to a fixed configuration, optimized for the first video. We add Model Reuse as a witness approach to measure how we can improve the standard performance model.

The **Model Reuse** selects a video of the training set, apply a model on it and keep a near-optimal configuration working for this video. Then, it applies this configuration to all inputs of the test set.


```python
MR_configs = np.loadtxt(res_dir+"MR_results.csv")
MR_ratios = [listVideoTest[i][predDimension][MR_configs[i]]/best_perfs[i] for i in range(len(listVideoTest))]
```

### Best compromise (BC)

**Best compromise (BC)** applies a performance model on all the training set, without making a difference between input videos. 
It selects the configuration working best for most videos in the training set. 
Technically, we rank the 201 configurations (1 being the optimal configuration, and 201 the worst) and select the one optimizing the sum of ranks for all input videos in the training set. 


```python
BC_configs = np.loadtxt(res_dir+"BC_results.csv")
BC_ratios = [listVideoTest[i][predDimension][BC_configs[i]]/best_perfs[i] for i in range(len(listVideoTest))]
```

### *Learning with properties*

### Direct Inclusion (DI)

**Direct Inclusion (DI)** includes input properties directly in the model during the training phase. The trained model then predicts the performance of x264 based on a set of properties (i.e. information about the input video) **and** a set of configuration options (i.e. information about the configuration). We fed this model with the 201 configurations of our dataset, and the properties of the test videos. We select the configuration giving the best prediction (e.g. the lowest bitrate).


```python
DI_configs = np.loadtxt(res_dir+"DI_results.csv")
DI_ratios = [listVideoTest[i][predDimension][DI_configs[i]]/best_perfs[i] for i in range(len(listVideoTest))]
```

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


```python
IaL_configs = np.loadtxt(res_dir+"IaL_results.csv")
IaL_ratios = [listVideoTest[i][predDimension][IaL_configs[i]]/best_perfs[i] for i in range(len(listVideoTest))]
```

### *Transfer Learning*

### Beetle


> @article{beetle,
  author    = {Rahul Krishna and
               Vivek Nair and
               Pooyan Jamshidi and
               Tim Menzies},
  title     = {Whence to Learn? Transferring Knowledge in Configurable Systems using
               {BEETLE}},
  journal   = {CoRR},
  volume    = {abs/1911.01817},
  year      = {2019},
  url       = {http://arxiv.org/abs/1911.01817},
  archivePrefix = {arXiv},
  eprint    = {1911.01817},
  timestamp = {Mon, 11 Nov 2019 18:38:09 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1911-01817.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

**Beetle** is a transfer learning approach defined by Krishna et al that relies on *source selection*. 
Given a (set of) input(s), the goal is to rank the sources by performance, in order to discover a bellwether input from which we can easily transfer performances (i.e. find the best source). 
Then, we transfer performances from this bellwether input to all inputs of the test set. 
We only apply the discovery phase (i.e. the search of bellwether) on the training set, to avoid introducing any bias in the results. 


```python
beetle_data = pd.read_csv(res_dir+"Beetle_results.csv").set_index("id_video")
beetle_data
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
      <th>conf5</th>
      <th>conf10</th>
      <th>conf15</th>
      <th>conf20</th>
      <th>conf25</th>
      <th>conf30</th>
    </tr>
    <tr>
      <th>id_video</th>
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
      <td>48</td>
      <td>112</td>
      <td>108</td>
      <td>104</td>
      <td>102</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>105</td>
      <td>108</td>
      <td>107</td>
      <td>105</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103</td>
      <td>102</td>
      <td>68</td>
      <td>189</td>
      <td>83</td>
      <td>200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>105</td>
      <td>62</td>
      <td>93</td>
      <td>107</td>
      <td>107</td>
      <td>112</td>
    </tr>
    <tr>
      <th>4</th>
      <td>143</td>
      <td>85</td>
      <td>107</td>
      <td>101</td>
      <td>103</td>
      <td>109</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>342</th>
      <td>100</td>
      <td>103</td>
      <td>102</td>
      <td>109</td>
      <td>100</td>
      <td>101</td>
    </tr>
    <tr>
      <th>343</th>
      <td>105</td>
      <td>53</td>
      <td>101</td>
      <td>93</td>
      <td>104</td>
      <td>85</td>
    </tr>
    <tr>
      <th>344</th>
      <td>41</td>
      <td>54</td>
      <td>93</td>
      <td>105</td>
      <td>101</td>
      <td>112</td>
    </tr>
    <tr>
      <th>345</th>
      <td>91</td>
      <td>47</td>
      <td>98</td>
      <td>103</td>
      <td>103</td>
      <td>105</td>
    </tr>
    <tr>
      <th>346</th>
      <td>185</td>
      <td>93</td>
      <td>64</td>
      <td>83</td>
      <td>105</td>
      <td>108</td>
    </tr>
  </tbody>
</table>
<p>347 rows × 6 columns</p>
</div>



### L2S

>@inproceedings{jamshidi2018,
    title={Learning to sample: exploiting similarities across environments to learn performance models for configurable systems}, 
    author={Jamshidi, Pooyan and Velez, Miguel and K{\"a}stner, Christian and Siegmund, Norbert},
    booktitle={Proceedings of the 2018 26th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
    pages={71--82},
    year={2018},
    organization={ACM},
    url={https://dl.acm.org/doi/pdf/10.1145/3236024.3236074},
}

**Learning to Sample (L2S)** is a transfer learning approach defined by Jamshidi et al. 
First, it exploits the source input and selects configurations that leverage influential (interactions of) features for this input. 
Then, it explores the similarities between the source and the target, thus adding configurations having similar performances for the source and the target. 
Finally, it uses the configurations selected in previous steps to efficiently train a model on the target input. 


```python
l2s_data = pd.read_csv(res_dir+"L2S_results.csv").set_index("id_video")
l2s_data
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
      <th>conf5</th>
      <th>conf10</th>
      <th>conf15</th>
      <th>conf20</th>
      <th>conf25</th>
      <th>conf30</th>
    </tr>
    <tr>
      <th>id_video</th>
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
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>100</td>
      <td>41</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>7</td>
      <td>83</td>
      <td>85</td>
      <td>42</td>
      <td>103</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>12</td>
      <td>100</td>
      <td>43</td>
      <td>95</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41</td>
      <td>4</td>
      <td>6</td>
      <td>98</td>
      <td>91</td>
      <td>97</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>1</td>
      <td>10</td>
      <td>100</td>
      <td>62</td>
      <td>25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>342</th>
      <td>4</td>
      <td>1</td>
      <td>42</td>
      <td>4</td>
      <td>104</td>
      <td>103</td>
    </tr>
    <tr>
      <th>343</th>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>100</td>
      <td>108</td>
      <td>105</td>
    </tr>
    <tr>
      <th>344</th>
      <td>6</td>
      <td>4</td>
      <td>105</td>
      <td>4</td>
      <td>46</td>
      <td>102</td>
    </tr>
    <tr>
      <th>345</th>
      <td>0</td>
      <td>15</td>
      <td>41</td>
      <td>41</td>
      <td>103</td>
      <td>41</td>
    </tr>
    <tr>
      <th>346</th>
      <td>4</td>
      <td>100</td>
      <td>20</td>
      <td>41</td>
      <td>100</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
<p>347 rows × 6 columns</p>
</div>



### Model Shift

>@inproceedings{DBLP:conf/wosp/ValovPGFC17,
  author    = {Pavel Valov and
               Jean{-}Christophe Petkovich and
               Jianmei Guo and
               Sebastian Fischmeister and
               Krzysztof Czarnecki},
  title     = {Transferring Performance Prediction Models Across Different Hardware
               Platforms},
  booktitle = {Proceedings of the 8th {ACM/SPEC} on International Conference on Performance
               Engineering, {ICPE} 2017, L'Aquila, Italy, April 22-26, 2017},
  pages     = {39--50},
  year      = {2017},
  url       = {http://doi.acm.org/10.1145/3030207.3030216},
  doi       = {10.1145/3030207.3030216},
  timestamp = {Sat, 22 Apr 2017 15:59:26 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/conf/wosp/ValovPGFC17},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}

**Model Shift (MS)** is a transfer learning defined by Valov et al. 
First, it trains a performance model on the source input and predicts the performance distribution of the source input. 
Then, it trains a *shifting function*, predicting the performances of the target input based on the performances of the source. 
Finally, it applies the shifting function to the predictions of the source. 
The original paper uses a linear function to transfer the performances between the source and the target. 
However, we extended this definition to any function (\eg random forest, neural network, \etc) able to learn the differences between the source and the target measurements. 


```python
ms_data = pd.read_csv(res_dir+"MS_results.csv").set_index("id_video")
ms_data
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
      <th>conf5</th>
      <th>conf10</th>
      <th>conf15</th>
      <th>conf20</th>
      <th>conf25</th>
      <th>conf30</th>
    </tr>
    <tr>
      <th>id_video</th>
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
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>42</td>
      <td>100</td>
      <td>102</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>102</td>
      <td>101</td>
      <td>41</td>
      <td>98</td>
      <td>41</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>4</td>
      <td>42</td>
      <td>41</td>
      <td>41</td>
      <td>46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>20</td>
      <td>42</td>
      <td>106</td>
      <td>42</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>41</td>
      <td>41</td>
      <td>94</td>
      <td>41</td>
      <td>98</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>342</th>
      <td>54</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
    </tr>
    <tr>
      <th>343</th>
      <td>25</td>
      <td>42</td>
      <td>41</td>
      <td>41</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>344</th>
      <td>1</td>
      <td>41</td>
      <td>48</td>
      <td>46</td>
      <td>48</td>
      <td>48</td>
    </tr>
    <tr>
      <th>345</th>
      <td>42</td>
      <td>4</td>
      <td>42</td>
      <td>42</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>346</th>
      <td>41</td>
      <td>47</td>
      <td>42</td>
      <td>103</td>
      <td>100</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
<p>347 rows × 6 columns</p>
</div>



### No Transfer

**No Transfer (NT)** is a Simple Learning approach, acting as a control approach to state whether transfer learning is suited to solve this problem. 
It trains a performance model directly on the target input, without using the source. 
We expect to outperform No Transfer with transfer learning approaches. 


```python
nt_data = pd.read_csv(res_dir+"NT_results.csv").set_index("id_video")
nt_data
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
      <th>conf5</th>
      <th>conf10</th>
      <th>conf15</th>
      <th>conf20</th>
      <th>conf25</th>
      <th>conf30</th>
    </tr>
    <tr>
      <th>id_video</th>
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
      <td>95</td>
      <td>4</td>
      <td>41</td>
      <td>100</td>
      <td>42</td>
      <td>200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>98</td>
      <td>85</td>
      <td>86</td>
      <td>109</td>
      <td>41</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>45</td>
      <td>46</td>
      <td>41</td>
      <td>41</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>63</td>
      <td>38</td>
      <td>91</td>
      <td>46</td>
      <td>41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>63</td>
      <td>85</td>
      <td>42</td>
      <td>62</td>
      <td>104</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>342</th>
      <td>46</td>
      <td>100</td>
      <td>42</td>
      <td>17</td>
      <td>100</td>
      <td>41</td>
    </tr>
    <tr>
      <th>343</th>
      <td>42</td>
      <td>69</td>
      <td>83</td>
      <td>85</td>
      <td>100</td>
      <td>41</td>
    </tr>
    <tr>
      <th>344</th>
      <td>41</td>
      <td>96</td>
      <td>100</td>
      <td>49</td>
      <td>48</td>
      <td>100</td>
    </tr>
    <tr>
      <th>345</th>
      <td>4</td>
      <td>71</td>
      <td>79</td>
      <td>200</td>
      <td>42</td>
      <td>98</td>
    </tr>
    <tr>
      <th>346</th>
      <td>41</td>
      <td>100</td>
      <td>102</td>
      <td>100</td>
      <td>23</td>
      <td>62</td>
    </tr>
  </tbody>
</table>
<p>347 rows × 6 columns</p>
</div>



## First results about properties - Figure 5a


```python
# we aggregate the different ratios, sorted by increasing efficiency
box_res = np.transpose(pd.DataFrame({"MR" : MR_ratios,
                                     "BC" : BC_ratios,
                                     "DI" : DI_ratios,
                                     "IaL" : IaL_ratios}))

# rotation of the text in the ordered axis, to fit the figure in the paper
degrees = 20

# cosmetic choices
red_square = dict(markerfacecolor='r', marker='s')
# figure size
plt.figure(figsize=(16,12))
# add a grid
plt.grid(alpha =0.5)
plt.boxplot(box_res, flierprops=red_square, 
          vert=True, patch_artist=True, widths=0.25,
          boxprops=dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
          whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))
# add crosses for average values of distributions
plt.scatter(np.arange(1, 5, 1), np.array([np.mean(box_res.iloc[i]) for i in range(4)]), 
            marker="x", color = "black", alpha = 1, s = 100)
# Limits
plt.ylim(0.9,2.5)
plt.xlim(0.5,4.5)
# Inputec vs Baseline
plt.vlines(x=2.5, ymin=0.5, ymax=4.5, linestyle='-.', color='green', linewidth = 5)
plt.text(s = "Simple Learning", x = 1.12, y = 2.25, size = 20, color = 'green')
plt.text(s = "Learning", x = 3.32, y = 2.3, size = 20, color = 'green')
plt.text(s = "with properties", x = 3.13, y = 2.2, size = 20, color = 'green')
# Labels
plt.ylabel("Ratio performance/best", size = 20)
plt.xlabel("", size = 20)
# Arrow
plt.arrow(x = 4.35, y = 2, dx= 0, dy = -0.3, head_width = 0.1, head_length = .1, color="orange")
plt.text(s = "Better", x = 4.05, y = 1.82, size = 20, color = 'orange')
# Ticks of axis
plt.xticks([1, 2, 3, 4], 
           ['Model Reuse (MR)', 'Best Compromise (BC)', 
            'Direct Inclusion (DI)', 'Input-aware Learning (IaL)'],
          size = 20,
          rotation=degrees)
plt.yticks(size = 20)
# save figure in the results folder
plt.savefig("../../results/res_box_approach.png")
# show the figure
plt.show()
```


    
![png](fps_files/fps_85_0.png)
    


#### Description of the results

##### Average results


```python
np.mean(BC_ratios)
```




    1.1963765688049417




```python
np.mean(MR_ratios)
```




    1.1277368506527417




```python
np.mean(IaL_ratios)
```




    1.1655585346035846




```python
np.mean(DI_ratios)
```




    1.244971983913473



##### Median results


```python
np.median(BC_ratios)
```




    1.1383508577753183




```python
np.median(MR_ratios)
```




    1.064128256513026




```python
np.median(IaL_ratios)
```




    1.1150614091790563




```python
np.median(DI_ratios)
```




    1.1614987080103358



##### IQR


```python
def iqr(distrib):
    return np.percentile(distrib, 75) - np.percentile(distrib, 25)
```


```python
iqr(MR_ratios)
```




    0.18368895847817424




```python
iqr(BC_ratios)
```




    0.17844191176362267




```python
iqr(IaL_ratios)
```




    0.17824876922538602




```python
iqr(DI_ratios)
```




    0.31351719565382874



#### Statistical tests - Welch t-test


```python
stats.wilcoxon(IaL_ratios, BC_ratios)
```




    WilcoxonResult(statistic=20552.5, pvalue=2.6519224267130245e-05)




```python
stats.wilcoxon(IaL_ratios, MR_ratios)
```




    WilcoxonResult(statistic=9901.0, pvalue=2.1883798583725074e-08)




```python
stats.wilcoxon(DI_ratios, BC_ratios)
```




    WilcoxonResult(statistic=21807.5, pvalue=0.00035743730223744937)




```python
stats.wilcoxon(DI_ratios, MR_ratios)
```




    WilcoxonResult(statistic=10275.0, pvalue=3.985087818100381e-17)



DI and MR, DI and BC, IaL and MR, IaL and BC are significantly different


```python
stats.wilcoxon(DI_ratios, IaL_ratios)
```




    WilcoxonResult(statistic=18045.0, pvalue=9.281499003583958e-08)



IaL and DI are significantly different


```python
stats.wilcoxon(BC_ratios, MR_ratios)
```




    WilcoxonResult(statistic=8092.5, pvalue=3.573785809450277e-22)



MR and BC are not significantly different

## Results about cost - Figure 5b

Aggregation of data


```python
f = []

cols = ["conf5", "conf10", "conf15", "conf20", "conf25", "conf30"]
names_cols = ['05', '10', '15', '20', '25', '30']
for i in range(len(listVideoTest)):
    for j in range(len(cols)):
        c = cols[j]
        nc = names_cols[j]
        f.append((listVideoTest[i][predDimension][beetle_data[c].iloc[i]]/best_perfs[i], nc, "Beetle"))
        f.append((listVideoTest[i][predDimension][ms_data[c].iloc[i]]/best_perfs[i], nc, "Model Shift (MS)"))
        f.append((listVideoTest[i][predDimension][nt_data[c].iloc[i]]/best_perfs[i], nc, "No Transfer (NT)"))
        f.append((listVideoTest[i][predDimension][l2s_data[c].iloc[i]]/best_perfs[i], nc, "Learning to Sample (L2S)"))


final_tl_data = pd.DataFrame(f, columns = ["ratio", "training_size", "Approach"])
final_tl_data
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
      <th>ratio</th>
      <th>training_size</th>
      <th>Approach</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.775216</td>
      <td>05</td>
      <td>Beetle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.734150</td>
      <td>05</td>
      <td>Model Shift (MS)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.565562</td>
      <td>05</td>
      <td>No Transfer (NT)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.301873</td>
      <td>05</td>
      <td>Learning to Sample (L2S)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.235591</td>
      <td>10</td>
      <td>Beetle</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8323</th>
      <td>1.000000</td>
      <td>25</td>
      <td>Learning to Sample (L2S)</td>
    </tr>
    <tr>
      <th>8324</th>
      <td>1.090595</td>
      <td>30</td>
      <td>Beetle</td>
    </tr>
    <tr>
      <th>8325</th>
      <td>1.149378</td>
      <td>30</td>
      <td>Model Shift (MS)</td>
    </tr>
    <tr>
      <th>8326</th>
      <td>3.558783</td>
      <td>30</td>
      <td>No Transfer (NT)</td>
    </tr>
    <tr>
      <th>8327</th>
      <td>1.000000</td>
      <td>30</td>
      <td>Learning to Sample (L2S)</td>
    </tr>
  </tbody>
</table>
<p>8328 rows × 3 columns</p>
</div>




```python
plt.figure(figsize=(16,12))

plt.grid(alpha=0.5)

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="training_size", y="ratio",
            hue="Approach", palette=["lightgreen", "coral", "lightgray", "purple"],
            data=final_tl_data, 
            showmeans=True, 
            meanprops={"marker":"x", "markeredgecolor":"black"})
plt.ylabel("Ratio performance/best", size = 20)
plt.xlabel("Budget - # Training configurations - Target", size = 20)
plt.ylim(0.9,15)
plt.legend(fontsize=20, loc = 'upper right', framealpha=1)


# Arrow
plt.arrow(x = 4.5, y = 2, dx= 0, dy = -0.3, head_width = 0.15, head_length = .1, color="orange")
plt.text(s = "Better", x = 4, y = 1.95, size = 20, color = 'orange')

plt.xticks(size = 20)
plt.yticks(size = 20)
plt.savefig("../../results/res_box_tl_approach.png")
plt.show()
```


    
![png](fps_files/fps_116_0.png)
    



```python
approaches = ["Beetle", "Learning to Sample (L2S)", "Model Shift (MS)", "No Transfer (NT)"]
budgets = np.arange(5, 31, 5)

def get_ratio(approach, training_size):
    if training_size < 10:
        training_size= '0'+str(training_size)
    return final_tl_data.query("Approach=='"+approach
                               +"' and training_size=='"+str(training_size)+"'")['ratio']
```


```python
np.mean(get_ratio("Model Shift (MS)", 10))
```




    3.256985395909188




```python
np.median(get_ratio("Model Shift (MS)", 10))
```




    2.00576




```python
iqr(get_ratio("Model Shift (MS)", 10))
```




    1.2447781610094135




```python
np.mean(get_ratio("Beetle", 10))
```




    2.5622087369670967




```python
np.median(get_ratio("Beetle", 10))
```




    1.5700757575757573




```python
iqr(get_ratio("Beetle", 10))
```




    1.6988325295381541




```python
np.median(get_ratio("Learning to Sample (L2S)", 10))
```




    3.8410596026490063




```python
stats.wilcoxon(DI_ratios, get_ratio("Learning to Sample (L2S)", 30))
```




    WilcoxonResult(statistic=9105.0, pvalue=7.348846646708823e-27)




```python
np.mean(get_ratio("Learning to Sample (L2S)", 30))
```




    2.2501330365954457




```python
np.median(get_ratio("Learning to Sample (L2S)", 30))
```




    1.5820224719101124




```python
np.mean(get_ratio("Learning to Sample (L2S)", 30)==1)
```




    0.06628242074927954




```python
iqr(get_ratio("Learning to Sample (L2S)", 10))
```




    7.4416294451601885




```python
iqr(get_ratio("Learning to Sample (L2S)", 30))
```




    1.2065565902528874




```python
iqr(get_ratio("Beetle", 5))
```




    4.121349019500986




```python
iqr(get_ratio("Beetle", 30))
```




    0.26206448233035884




```python
iqr(get_ratio("Model Shift (MS)", 10))
```




    1.2447781610094135




```python
iqr(get_ratio("Model Shift (MS)", 30))
```




    0.7729909244839046




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
