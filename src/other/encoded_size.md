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
predDimension = "size"

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
    mean          0.566601
    std           0.288890
    min          -0.693836
    25%           0.394660
    50%           0.630341
    75%           0.794691
    max           0.997228
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

    Value :  -0.6938364419242731
    i :  9 , j :  1166



```python
corrSpearman[378][1192]
```




    0.41941853032782944




```python
corrSpearman[314][1192]
```




    0.21037172235174148




```python
corrSpearman[378][314]
```




    0.8963699710706091



#### "For 95% of the videos, it is always possible to find another video having a correlation higher than 0.92" -> here is the proof 


```python
argm = [np.max([k for k in corrSpearman[i] if k <1]) for i in range(len(corrSpearman))]
pd.Series(argm).describe()
```




    count    1397.000000
    mean        0.969512
    std         0.022921
    min         0.803562
    25%         0.963100
    50%         0.975897
    75%         0.983899
    max         0.997228
    dtype: float64




```python
np.percentile(argm, 5)
```




    0.9216475343513424



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


    
![png](encoded_size_files/encoded_size_28_0.png)
    


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

    Group 1 contains 470 input videos.
    Group 2 contains 219 input videos.
    Group 3 contains 262 input videos.
    Group 4 contains 446 input videos.



```python
[v_names[i] for i in np.where(groups==3)[0]]
```




    ['Animation_1080P-0c4f.csv',
     'Animation_1080P-3dbf.csv',
     'Animation_1080P-3e01.csv',
     'Animation_1080P-646f.csv',
     'Animation_360P-08c9.csv',
     'Animation_360P-188f.csv',
     'Animation_360P-24d4.csv',
     'Animation_360P-47cc.csv',
     'Animation_360P-4b4c.csv',
     'Animation_360P-4edc.csv',
     'Animation_360P-5712.csv',
     'Animation_360P-5de0.csv',
     'Animation_360P-631c.csv',
     'Animation_360P-69e0.csv',
     'Animation_360P-794f.csv',
     'Animation_480P-046c.csv',
     'Animation_480P-0529.csv',
     'Animation_480P-35ee.csv',
     'Animation_480P-4b80.csv',
     'Animation_480P-4e36.csv',
     'Animation_480P-6e23.csv',
     'Animation_480P-6ef6.csv',
     'Animation_720P-0116.csv',
     'Animation_720P-1a6d.csv',
     'Animation_720P-4268.csv',
     'Animation_720P-57d9.csv',
     'Animation_720P-79ee.csv',
     'Animation_720P-7b29.csv',
     'CoverSong_1080P-1b08.csv',
     'CoverSong_1080P-3409.csv',
     'CoverSong_1080P-3aac.csv',
     'CoverSong_1080P-3df8.csv',
     'CoverSong_1080P-5430.csv',
     'CoverSong_360P-11f9.csv',
     'CoverSong_360P-1b2b.csv',
     'CoverSong_360P-2146.csv',
     'CoverSong_360P-2b4d.csv',
     'CoverSong_360P-53a6.csv',
     'CoverSong_360P-5a24.csv',
     'CoverSong_480P-1019.csv',
     'CoverSong_480P-4d34.csv',
     'CoverSong_480P-53f4.csv',
     'CoverSong_480P-59f2.csv',
     'CoverSong_480P-5b62.csv',
     'CoverSong_480P-60a6.csv',
     'CoverSong_480P-64d0.csv',
     'CoverSong_480P-6c3e.csv',
     'CoverSong_480P-7443.csv',
     'CoverSong_720P-01a1.csv',
     'CoverSong_720P-449f.csv',
     'Gaming_1080P-0a5b.csv',
     'Gaming_1080P-13e3.csv',
     'Gaming_1080P-277c.csv',
     'Gaming_1080P-2927.csv',
     'Gaming_1080P-2e97.csv',
     'Gaming_1080P-3a9d.csv',
     'Gaming_1080P-51fc.csv',
     'Gaming_1080P-6578.csv',
     'Gaming_1080P-698a.csv',
     'Gaming_1080P-6d53.csv',
     'Gaming_1080P-6db2.csv',
     'Gaming_1080P-7a1e.csv',
     'Gaming_2160P-31f6.csv',
     'Gaming_2160P-348d.csv',
     'Gaming_2160P-3aec.csv',
     'Gaming_360P-043e.csv',
     'Gaming_360P-187a.csv',
     'Gaming_360P-21d2.csv',
     'Gaming_360P-2330.csv',
     'Gaming_360P-3794.csv',
     'Gaming_360P-3eb6.csv',
     'Gaming_360P-48b0.csv',
     'Gaming_360P-586d.csv',
     'Gaming_360P-5e0f.csv',
     'Gaming_360P-7007.csv',
     'Gaming_360P-7975.csv',
     'Gaming_360P-7acb.csv',
     'Gaming_480P-0a30.csv',
     'Gaming_480P-445b.csv',
     'Gaming_480P-4560.csv',
     'Gaming_480P-5a5a.csv',
     'Gaming_480P-61ee.csv',
     'Gaming_480P-626a.csv',
     'Gaming_480P-6a5a.csv',
     'Gaming_480P-6c92.csv',
     'Gaming_480P-6d1e.csv',
     'Gaming_480P-6f4b.csv',
     'Gaming_480P-75f7.csv',
     'Gaming_480P-7a08.csv',
     'Gaming_480P-7ccb.csv',
     'Gaming_720P-0fba.csv',
     'Gaming_720P-221d.csv',
     'Gaming_720P-25aa.csv',
     'Gaming_720P-2dbe.csv',
     'Gaming_720P-312f.csv',
     'Gaming_720P-3524.csv',
     'Gaming_720P-493e.csv',
     'Gaming_720P-4cda.csv',
     'Gaming_720P-6403.csv',
     'Gaming_720P-6625.csv',
     'Gaming_720P-6658.csv',
     'Gaming_720P-6a45.csv',
     'HDR_1080P-2d32.csv',
     'HDR_1080P-4f4a.csv',
     'HDR_1080P-69de.csv',
     'HDR_1080P-7825.csv',
     'HDR_2160P-1743.csv',
     'HDR_2160P-2a72.csv',
     'HDR_2160P-382f.csv',
     'HDR_2160P-3bf1.csv',
     'HDR_2160P-5e25.csv',
     'HDR_2160P-6fab.csv',
     'HDR_2160P-70ca.csv',
     'HowTo_1080P-0267.csv',
     'HowTo_1080P-13aa.csv',
     'HowTo_1080P-36a9.csv',
     'HowTo_1080P-52bb.csv',
     'HowTo_1080P-55d1.csv',
     'HowTo_1080P-63e4.csv',
     'HowTo_1080P-64f7.csv',
     'HowTo_1080P-7cf2.csv',
     'HowTo_360P-041c.csv',
     'HowTo_360P-1823.csv',
     'HowTo_360P-3aa6.csv',
     'HowTo_360P-6a0e.csv',
     'HowTo_480P-0eb3.csv',
     'HowTo_480P-4332.csv',
     'HowTo_480P-470b.csv',
     'HowTo_480P-4a28.csv',
     'HowTo_480P-4c99.csv',
     'HowTo_480P-63a2.csv',
     'HowTo_480P-7579.csv',
     'HowTo_480P-7c11.csv',
     'HowTo_720P-269e.csv',
     'HowTo_720P-37d0.csv',
     'HowTo_720P-3a5d.csv',
     'HowTo_720P-6323.csv',
     'HowTo_720P-6791.csv',
     'HowTo_720P-7878.csv',
     'HowTo_720P-7c38.csv',
     'Lecture_1080P-011f.csv',
     'Lecture_1080P-1709.csv',
     'Lecture_1080P-238b.csv',
     'Lecture_1080P-3ce0.csv',
     'Lecture_1080P-3cf3.csv',
     'Lecture_1080P-4991.csv',
     'Lecture_1080P-6089.csv',
     'Lecture_1080P-73d5.csv',
     'Lecture_360P-20c3.csv',
     'Lecture_360P-27db.csv',
     'Lecture_360P-30eb.csv',
     'Lecture_360P-311d.csv',
     'Lecture_360P-4f00.csv',
     'Lecture_360P-7550.csv',
     'Lecture_360P-7f7e.csv',
     'Lecture_480P-11df.csv',
     'Lecture_480P-2513.csv',
     'Lecture_480P-2655.csv',
     'Lecture_480P-369f.csv',
     'Lecture_480P-41b7.csv',
     'Lecture_480P-5cd7.csv',
     'Lecture_480P-5f3a.csv',
     'Lecture_480P-71c0.csv',
     'Lecture_480P-71d6.csv',
     'Lecture_480P-7d77.csv',
     'Lecture_720P-003a.csv',
     'Lecture_720P-07e0.csv',
     'Lecture_720P-10bc.csv',
     'Lecture_720P-1f22.csv',
     'Lecture_720P-2f38.csv',
     'Lecture_720P-50b9.csv',
     'Lecture_720P-5120.csv',
     'Lecture_720P-5725.csv',
     'LiveMusic_1080P-14af.csv',
     'LiveMusic_1080P-157b.csv',
     'LiveMusic_1080P-2b7a.csv',
     'LiveMusic_1080P-2f7f.csv',
     'LiveMusic_1080P-3f95.csv',
     'LiveMusic_1080P-51f6.csv',
     'LiveMusic_1080P-6b1c.csv',
     'LiveMusic_1080P-6bbe.csv',
     'LiveMusic_1080P-7948.csv',
     'LiveMusic_360P-1d94.csv',
     'LiveMusic_360P-22c5.csv',
     'LiveMusic_360P-405c.csv',
     'LiveMusic_360P-6266.csv',
     'LiveMusic_360P-6640.csv',
     'LiveMusic_360P-7483.csv',
     'LiveMusic_480P-2019.csv',
     'LiveMusic_480P-331a.csv',
     'LiveMusic_480P-34be.csv',
     'LiveMusic_480P-4c3a.csv',
     'LiveMusic_480P-559d.csv',
     'LiveMusic_480P-58fb.csv',
     'LiveMusic_480P-61ef.csv',
     'LiveMusic_480P-636e.csv',
     'LiveMusic_480P-65ca.csv',
     'LiveMusic_720P-3320.csv',
     'LiveMusic_720P-6343.csv',
     'LiveMusic_720P-6452.csv',
     'LiveMusic_720P-66df.csv',
     'LiveMusic_720P-71c5.csv',
     'LyricVideo_1080P-0625.csv',
     'LyricVideo_1080P-12af.csv',
     'LyricVideo_1080P-16b6.csv',
     'LyricVideo_1080P-584f.csv',
     'LyricVideo_360P-5e87.csv',
     'LyricVideo_480P-0722.csv',
     'LyricVideo_480P-2c50.csv',
     'LyricVideo_480P-3ccf.csv',
     'LyricVideo_480P-4346.csv',
     'LyricVideo_480P-5c17.csv',
     'LyricVideo_480P-6385.csv',
     'LyricVideo_720P-068d.csv',
     'LyricVideo_720P-0940.csv',
     'LyricVideo_720P-2d24.csv',
     'LyricVideo_720P-47a9.csv',
     'LyricVideo_720P-59ed.csv',
     'LyricVideo_720P-6f0c.csv',
     'LyricVideo_720P-74a0.csv',
     'MusicVideo_1080P-04b6.csv',
     'MusicVideo_1080P-0706.csv',
     'MusicVideo_1080P-106d.csv',
     'MusicVideo_1080P-16e6.csv',
     'MusicVideo_1080P-453f.csv',
     'MusicVideo_1080P-4671.csv',
     'MusicVideo_1080P-55af.csv',
     'MusicVideo_1080P-7f2e.csv',
     'MusicVideo_360P-17e4.csv',
     'MusicVideo_360P-2fcb.csv',
     'MusicVideo_360P-5578.csv',
     'MusicVideo_360P-5f07.csv',
     'MusicVideo_360P-7b94.csv',
     'MusicVideo_480P-001f.csv',
     'MusicVideo_480P-184c.csv',
     'MusicVideo_480P-1eee.csv',
     'MusicVideo_480P-2de0.csv',
     'MusicVideo_480P-41ce.csv',
     'MusicVideo_480P-4802.csv',
     'MusicVideo_480P-483b.csv',
     'MusicVideo_480P-4cc8.csv',
     'MusicVideo_480P-6026.csv',
     'MusicVideo_480P-61ba.csv',
     'MusicVideo_480P-6fb6.csv',
     'MusicVideo_480P-7643.csv',
     'MusicVideo_720P-2d7d.csv',
     'MusicVideo_720P-3698.csv',
     'MusicVideo_720P-44c1.csv',
     'MusicVideo_720P-5c9c.csv',
     'MusicVideo_720P-62df.csv',
     'MusicVideo_720P-7501.csv',
     'NewsClip_1080P-06df.csv',
     'NewsClip_1080P-1db0.csv',
     'NewsClip_1080P-22b3.csv',
     'NewsClip_1080P-2eb0.csv',
     'NewsClip_1080P-3427.csv',
     'NewsClip_1080P-3c7c.csv',
     'NewsClip_1080P-5be1.csv',
     'NewsClip_360P-0376.csv',
     'NewsClip_360P-0ff8.csv',
     'NewsClip_360P-1093.csv',
     'NewsClip_360P-12fc.csv',
     'NewsClip_360P-1e1c.csv',
     'NewsClip_360P-2986.csv',
     'NewsClip_360P-311a.csv',
     'NewsClip_360P-439a.csv',
     'NewsClip_360P-5bcc.csv',
     'NewsClip_360P-67ce.csv',
     'NewsClip_480P-0269.csv',
     'NewsClip_480P-0ce5.csv',
     'NewsClip_480P-28eb.csv',
     'NewsClip_480P-41b1.csv',
     'NewsClip_480P-4a9f.csv',
     'NewsClip_480P-4e77.csv',
     'NewsClip_480P-543f.csv',
     'NewsClip_480P-696e.csv',
     'NewsClip_480P-7a0d.csv',
     'NewsClip_720P-04ba.csv',
     'NewsClip_720P-0c81.csv',
     'NewsClip_720P-5787.csv',
     'NewsClip_720P-6a19.csv',
     'NewsClip_720P-6aa6.csv',
     'NewsClip_720P-739b.csv',
     'Sports_1080P-28a6.csv',
     'Sports_1080P-2a21.csv',
     'Sports_1080P-3db7.csv',
     'Sports_1080P-43e2.csv',
     'Sports_1080P-4978.csv',
     'Sports_1080P-6571.csv',
     'Sports_1080P-6710.csv',
     'Sports_1080P-7203.csv',
     'Sports_1080P-76a2.csv',
     'Sports_2160P-1b70.csv',
     'Sports_2160P-2626.csv',
     'Sports_2160P-279f.csv',
     'Sports_2160P-2a83.csv',
     'Sports_2160P-2e1d.csv',
     'Sports_2160P-300d.csv',
     'Sports_2160P-3a9a.csv',
     'Sports_2160P-49f1.csv',
     'Sports_2160P-7165.csv',
     'Sports_360P-0dda.csv',
     'Sports_360P-11b7.csv',
     'Sports_360P-2ace.csv',
     'Sports_360P-3960.csv',
     'Sports_360P-3e68.csv',
     'Sports_360P-4ad7.csv',
     'Sports_360P-5252.csv',
     'Sports_360P-5ded.csv',
     'Sports_360P-61f6.csv',
     'Sports_360P-6f62.csv',
     'Sports_480P-1019.csv',
     'Sports_480P-2dfe.csv',
     'Sports_480P-3404.csv',
     'Sports_480P-5224.csv',
     'Sports_480P-6e41.csv',
     'Sports_480P-7f7e.csv',
     'Sports_720P-2191.csv',
     'Sports_720P-2632.csv',
     'Sports_720P-33c6.csv',
     'Sports_720P-531c.csv',
     'Sports_720P-5833.csv',
     'Sports_720P-5ae1.csv',
     'Sports_720P-5e39.csv',
     'Sports_720P-69a0.csv',
     'Sports_720P-6bb7.csv',
     'TelevisionClip_1080P-3d10.csv',
     'TelevisionClip_1080P-401e.csv',
     'TelevisionClip_1080P-5278.csv',
     'TelevisionClip_1080P-63e6.csv',
     'TelevisionClip_1080P-64e2.csv',
     'TelevisionClip_1080P-7eff.csv',
     'TelevisionClip_480P-09d8.csv',
     'TelevisionClip_480P-0e46.csv',
     'TelevisionClip_480P-27ca.csv',
     'TelevisionClip_480P-2ead.csv',
     'TelevisionClip_480P-3284.csv',
     'TelevisionClip_480P-3617.csv',
     'TelevisionClip_480P-373d.csv',
     'TelevisionClip_480P-436c.csv',
     'TelevisionClip_480P-59f0.csv',
     'TelevisionClip_480P-723e.csv',
     'TelevisionClip_720P-2b20.csv',
     'TelevisionClip_720P-31ce.csv',
     'TelevisionClip_720P-4af1.csv',
     'VR_1080P-0427.csv',
     'VR_1080P-08f0.csv',
     'VR_1080P-1a9c.csv',
     'VR_1080P-1d16.csv',
     'VR_1080P-2831.csv',
     'VR_1080P-2f02.csv',
     'VR_1080P-38d6.csv',
     'VR_1080P-3f81.csv',
     'VR_1080P-6261.csv',
     'VR_1080P-6501.csv',
     'VR_1080P-7067.csv',
     'VR_1080P-76a2.csv',
     'VR_2160P-1422.csv',
     'VR_2160P-1456.csv',
     'VR_2160P-29c3.csv',
     'VR_2160P-40af.csv',
     'VR_2160P-5c23.csv',
     'VR_2160P-6d71.csv',
     'VR_720P-00d2.csv',
     'VR_720P-063e.csv',
     'VR_720P-09ae.csv',
     'VR_720P-1b9a.csv',
     'VR_720P-1dd7.csv',
     'VR_720P-32ed.csv',
     'VR_720P-3386.csv',
     'VR_720P-339f.csv',
     'VR_720P-364f.csv',
     'VR_720P-403e.csv',
     'VR_720P-448a.csv',
     'VR_720P-557f.csv',
     'VR_720P-748e.csv',
     'VerticalVideo_1080P-04d4.csv',
     'VerticalVideo_1080P-1105.csv',
     'VerticalVideo_1080P-1ac1.csv',
     'VerticalVideo_1080P-34ba.csv',
     'VerticalVideo_1080P-3709.csv',
     'VerticalVideo_1080P-3a9b.csv',
     'VerticalVideo_1080P-3d96.csv',
     'VerticalVideo_360P-2fa3.csv',
     'VerticalVideo_360P-3936.csv',
     'VerticalVideo_360P-54f7.csv',
     'VerticalVideo_360P-579c.csv',
     'VerticalVideo_360P-634f.csv',
     'VerticalVideo_360P-6490.csv',
     'VerticalVideo_360P-694d.csv',
     'VerticalVideo_360P-7ec3.csv',
     'VerticalVideo_480P-2aa1.csv',
     'VerticalVideo_480P-419c.csv',
     'VerticalVideo_480P-51b7.csv',
     'VerticalVideo_720P-0750.csv',
     'VerticalVideo_720P-0f61.csv',
     'VerticalVideo_720P-2efc.csv',
     'VerticalVideo_720P-44de.csv',
     'VerticalVideo_720P-4730.csv',
     'VerticalVideo_720P-4ca7.csv',
     'VerticalVideo_720P-57eb.csv',
     'VerticalVideo_720P-7517.csv',
     'VerticalVideo_720P-7859.csv',
     'VerticalVideo_720P-7c1d.csv',
     'Vlog_1080P-04fe.csv',
     'Vlog_1080P-21f5.csv',
     'Vlog_1080P-37cf.csv',
     'Vlog_1080P-4ba9.csv',
     'Vlog_1080P-4f26.csv',
     'Vlog_1080P-52fe.csv',
     'Vlog_1080P-5904.csv',
     'Vlog_1080P-5a19.csv',
     'Vlog_1080P-634b.csv',
     'Vlog_1080P-6686.csv',
     'Vlog_1080P-7e8c.csv',
     'Vlog_2160P-0289.csv',
     'Vlog_2160P-030a.csv',
     'Vlog_2160P-1021.csv',
     'Vlog_2160P-19f9.csv',
     'Vlog_2160P-217c.csv',
     'Vlog_2160P-255c.csv',
     'Vlog_2160P-2b2d.csv',
     'Vlog_2160P-3019.csv',
     'Vlog_2160P-408f.csv',
     'Vlog_2160P-4362.csv',
     'Vlog_2160P-4655.csv',
     'Vlog_2160P-62b2.csv',
     'Vlog_2160P-6f92.csv',
     'Vlog_2160P-7b10.csv',
     'Vlog_2160P-7bfb.csv',
     'Vlog_360P-2973.csv',
     'Vlog_360P-2e9d.csv',
     'Vlog_360P-433e.csv',
     'Vlog_360P-4ad1.csv',
     'Vlog_360P-4d71.csv',
     'Vlog_360P-7334.csv',
     'Vlog_360P-76ae.csv',
     'Vlog_360P-7efe.csv',
     'Vlog_480P-08c7.csv',
     'Vlog_480P-1b39.csv',
     'Vlog_480P-59dc.csv',
     'Vlog_480P-5dfe.csv',
     'Vlog_480P-7754.csv',
     'Vlog_480P-7d0c.csv',
     'Vlog_720P-561e.csv',
     'Vlog_720P-6d56.csv']



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
      <td>166</td>
      <td>91</td>
      <td>104</td>
      <td>171</td>
      <td>169</td>
      <td>89</td>
      <td>26</td>
      <td>161</td>
      <td>171</td>
      <td>104</td>
      <td>...</td>
      <td>104</td>
      <td>169</td>
      <td>104</td>
      <td>164</td>
      <td>170</td>
      <td>169</td>
      <td>104</td>
      <td>170</td>
      <td>175</td>
      <td>169</td>
    </tr>
    <tr>
      <th>1</th>
      <td>164</td>
      <td>104</td>
      <td>105</td>
      <td>130</td>
      <td>168</td>
      <td>170</td>
      <td>27</td>
      <td>160</td>
      <td>32</td>
      <td>171</td>
      <td>...</td>
      <td>32</td>
      <td>166</td>
      <td>46</td>
      <td>175</td>
      <td>171</td>
      <td>168</td>
      <td>171</td>
      <td>171</td>
      <td>193</td>
      <td>168</td>
    </tr>
    <tr>
      <th>2</th>
      <td>169</td>
      <td>100</td>
      <td>100</td>
      <td>131</td>
      <td>170</td>
      <td>169</td>
      <td>28</td>
      <td>163</td>
      <td>161</td>
      <td>60</td>
      <td>...</td>
      <td>171</td>
      <td>170</td>
      <td>190</td>
      <td>193</td>
      <td>169</td>
      <td>165</td>
      <td>60</td>
      <td>169</td>
      <td>164</td>
      <td>123</td>
    </tr>
    <tr>
      <th>3</th>
      <td>165</td>
      <td>102</td>
      <td>102</td>
      <td>132</td>
      <td>165</td>
      <td>168</td>
      <td>21</td>
      <td>171</td>
      <td>160</td>
      <td>32</td>
      <td>...</td>
      <td>36</td>
      <td>165</td>
      <td>176</td>
      <td>92</td>
      <td>184</td>
      <td>123</td>
      <td>32</td>
      <td>60</td>
      <td>39</td>
      <td>159</td>
    </tr>
    <tr>
      <th>4</th>
      <td>168</td>
      <td>108</td>
      <td>108</td>
      <td>32</td>
      <td>123</td>
      <td>165</td>
      <td>20</td>
      <td>130</td>
      <td>163</td>
      <td>130</td>
      <td>...</td>
      <td>35</td>
      <td>168</td>
      <td>187</td>
      <td>112</td>
      <td>179</td>
      <td>159</td>
      <td>4</td>
      <td>177</td>
      <td>23</td>
      <td>170</td>
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

    Config min std :  200
    Config max std :  16
    Config med std :  58.678078128140996



    
![png](encoded_size_files/encoded_size_36_1.png)
    



    
![png](encoded_size_files/encoded_size_36_2.png)
    


#### Some statistics (not mentioned in the text)


```python
print("For config. 200, ", int(np.sum([1  for j in range(len(rankings.loc[np.argmin(stds),:])) 
              if rankings.loc[np.argmin(stds),:][j] > 105 and rankings.loc[np.argmin(stds),:][j] < 130])
      /len(rankings.loc[np.argmin(stds),:])*100),"% of configurations are between 105 and 130!")
```

    For config. 200,  92 % of configurations are between 105 and 130!



```python
np.where(rankings.loc[np.argmin(stds),:] == np.min(rankings.loc[np.argmin(stds),:]))
```




    (array([ 897, 1289, 1311, 1387]),)




```python
np.max(rankings.loc[np.argmax(stds),:])
```




    200




```python
np.where(rankings.loc[np.argmax(stds),:] == np.min(rankings.loc[np.argmax(stds),:]))
```




    (array([  11,  121,  698,  883, 1071, 1279]),)




```python
np.max(rankings.loc[np.argmax(stds),:])
```




    200




```python
np.where(rankings.loc[np.argmin(stds),:] == np.max(rankings.loc[np.argmin(stds),:]))
```




    (array([  15,   91,  132,  134,  316,  318,  390,  394,  402,  437,  503,
             507,  525,  535,  583,  585,  667,  696,  715,  927, 1025, 1046,
            1113, 1167, 1175, 1185, 1208, 1211, 1215, 1259, 1262, 1298, 1304,
            1312, 1318, 1322, 1380, 1390, 1395]),)



#### Rankings distributions


```python
pd.Series(rankings.loc[np.argmax(stds),:]).describe()
```




    count    1397.000000
    mean      107.672155
    std        62.910121
    min         2.000000
    25%        54.000000
    50%       103.000000
    75%       171.000000
    max       200.000000
    Name: 16, dtype: float64




```python
pd.Series(rankings.loc[np.argmin(stds),:]).describe()
```




    count    1397.000000
    mean      120.461704
    std        19.612164
    min         6.000000
    25%       113.000000
    50%       120.000000
    75%       124.000000
    max       199.000000
    Name: 200, dtype: float64



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


    
![png](encoded_size_files/encoded_size_52_0.png)
    


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
      <td>-0.18</td>
      <td>-0.00</td>
      <td>0.06</td>
      <td>-0.07</td>
      <td>-0.07</td>
      <td>-0.03</td>
      <td>-0.03</td>
      <td>-0.06</td>
      <td>-0.09</td>
      <td>-0.01</td>
      <td>...</td>
      <td>0.05</td>
      <td>-0.02</td>
      <td>-0.18</td>
      <td>0.0</td>
      <td>-0.13</td>
      <td>-0.15</td>
      <td>-0.01</td>
      <td>-0.05</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>video_1</th>
      <td>-0.25</td>
      <td>-0.15</td>
      <td>-0.04</td>
      <td>-0.04</td>
      <td>-0.17</td>
      <td>-0.28</td>
      <td>0.00</td>
      <td>-0.04</td>
      <td>0.04</td>
      <td>-0.19</td>
      <td>...</td>
      <td>0.04</td>
      <td>-0.01</td>
      <td>-0.14</td>
      <td>0.0</td>
      <td>-0.06</td>
      <td>0.03</td>
      <td>-0.01</td>
      <td>0.13</td>
      <td>0.0</td>
      <td>-0.43</td>
    </tr>
    <tr>
      <th>video_2</th>
      <td>-0.19</td>
      <td>0.00</td>
      <td>-0.05</td>
      <td>-0.03</td>
      <td>-0.07</td>
      <td>-0.30</td>
      <td>-0.05</td>
      <td>-0.03</td>
      <td>0.03</td>
      <td>-0.11</td>
      <td>...</td>
      <td>0.06</td>
      <td>-0.02</td>
      <td>-0.34</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>-0.04</td>
      <td>0.05</td>
      <td>-0.02</td>
      <td>0.0</td>
      <td>-0.27</td>
    </tr>
    <tr>
      <th>video_3</th>
      <td>-0.15</td>
      <td>0.07</td>
      <td>-0.09</td>
      <td>-0.04</td>
      <td>-0.06</td>
      <td>-0.32</td>
      <td>-0.01</td>
      <td>-0.05</td>
      <td>0.11</td>
      <td>-0.17</td>
      <td>...</td>
      <td>0.12</td>
      <td>-0.01</td>
      <td>-0.33</td>
      <td>0.0</td>
      <td>0.03</td>
      <td>-0.01</td>
      <td>0.05</td>
      <td>0.27</td>
      <td>0.0</td>
      <td>-0.43</td>
    </tr>
    <tr>
      <th>video_4</th>
      <td>-0.14</td>
      <td>0.02</td>
      <td>-0.05</td>
      <td>-0.00</td>
      <td>-0.10</td>
      <td>-0.26</td>
      <td>-0.01</td>
      <td>-0.04</td>
      <td>0.06</td>
      <td>-0.20</td>
      <td>...</td>
      <td>0.13</td>
      <td>-0.04</td>
      <td>-0.33</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.03</td>
      <td>0.11</td>
      <td>0.0</td>
      <td>-0.50</td>
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
      <td>-0.16</td>
      <td>0.05</td>
      <td>-0.09</td>
      <td>-0.05</td>
      <td>-0.06</td>
      <td>-0.33</td>
      <td>-0.01</td>
      <td>-0.06</td>
      <td>0.05</td>
      <td>-0.20</td>
      <td>...</td>
      <td>0.19</td>
      <td>-0.02</td>
      <td>-0.36</td>
      <td>0.0</td>
      <td>0.06</td>
      <td>-0.04</td>
      <td>0.11</td>
      <td>-0.13</td>
      <td>0.0</td>
      <td>-0.29</td>
    </tr>
    <tr>
      <th>video_1393</th>
      <td>-0.06</td>
      <td>0.11</td>
      <td>-0.13</td>
      <td>-0.08</td>
      <td>-0.05</td>
      <td>-0.37</td>
      <td>-0.02</td>
      <td>-0.06</td>
      <td>0.11</td>
      <td>-0.21</td>
      <td>...</td>
      <td>0.08</td>
      <td>-0.00</td>
      <td>-0.33</td>
      <td>0.0</td>
      <td>0.03</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>0.42</td>
      <td>0.0</td>
      <td>-0.47</td>
    </tr>
    <tr>
      <th>video_1394</th>
      <td>-0.14</td>
      <td>0.07</td>
      <td>-0.12</td>
      <td>-0.06</td>
      <td>-0.06</td>
      <td>-0.41</td>
      <td>-0.01</td>
      <td>-0.06</td>
      <td>0.10</td>
      <td>-0.21</td>
      <td>...</td>
      <td>0.16</td>
      <td>0.00</td>
      <td>-0.31</td>
      <td>0.0</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.09</td>
      <td>0.0</td>
      <td>-0.35</td>
    </tr>
    <tr>
      <th>video_1395</th>
      <td>-0.44</td>
      <td>0.01</td>
      <td>-0.07</td>
      <td>-0.05</td>
      <td>-0.10</td>
      <td>-0.12</td>
      <td>-0.00</td>
      <td>-0.02</td>
      <td>0.10</td>
      <td>-0.04</td>
      <td>...</td>
      <td>-0.06</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>-0.0</td>
      <td>0.03</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>-0.79</td>
      <td>0.0</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>video_1396</th>
      <td>-0.35</td>
      <td>0.05</td>
      <td>-0.11</td>
      <td>-0.04</td>
      <td>-0.08</td>
      <td>-0.36</td>
      <td>0.07</td>
      <td>-0.10</td>
      <td>0.20</td>
      <td>-0.21</td>
      <td>...</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>-0.14</td>
      <td>-0.0</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.06</td>
      <td>0.02</td>
      <td>0.0</td>
      <td>-0.25</td>
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


    
![png](encoded_size_files/encoded_size_56_0.png)
    


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
      <td>134</td>
      <td>105</td>
      <td>108</td>
      <td>107</td>
      <td>170</td>
      <td>123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>196</td>
      <td>169</td>
      <td>85</td>
      <td>91</td>
      <td>168</td>
      <td>173</td>
    </tr>
    <tr>
      <th>2</th>
      <td>74</td>
      <td>107</td>
      <td>172</td>
      <td>165</td>
      <td>85</td>
      <td>123</td>
    </tr>
    <tr>
      <th>3</th>
      <td>137</td>
      <td>91</td>
      <td>97</td>
      <td>168</td>
      <td>173</td>
      <td>196</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>189</td>
      <td>170</td>
      <td>91</td>
      <td>91</td>
      <td>169</td>
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
      <td>105</td>
      <td>91</td>
      <td>165</td>
      <td>168</td>
      <td>170</td>
      <td>165</td>
    </tr>
    <tr>
      <th>343</th>
      <td>173</td>
      <td>17</td>
      <td>168</td>
      <td>197</td>
      <td>170</td>
      <td>170</td>
    </tr>
    <tr>
      <th>344</th>
      <td>24</td>
      <td>159</td>
      <td>189</td>
      <td>102</td>
      <td>169</td>
      <td>196</td>
    </tr>
    <tr>
      <th>345</th>
      <td>74</td>
      <td>85</td>
      <td>96</td>
      <td>100</td>
      <td>91</td>
      <td>134</td>
    </tr>
    <tr>
      <th>346</th>
      <td>15</td>
      <td>100</td>
      <td>85</td>
      <td>134</td>
      <td>169</td>
      <td>91</td>
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
      <td>24</td>
      <td>182</td>
      <td>57</td>
      <td>168</td>
      <td>170</td>
      <td>154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99</td>
      <td>100</td>
      <td>102</td>
      <td>100</td>
      <td>165</td>
      <td>168</td>
    </tr>
    <tr>
      <th>2</th>
      <td>76</td>
      <td>195</td>
      <td>34</td>
      <td>58</td>
      <td>60</td>
      <td>161</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34</td>
      <td>56</td>
      <td>12</td>
      <td>36</td>
      <td>160</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>3</td>
      <td>97</td>
      <td>112</td>
      <td>108</td>
      <td>47</td>
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
      <td>1</td>
      <td>169</td>
      <td>169</td>
      <td>58</td>
      <td>36</td>
      <td>104</td>
    </tr>
    <tr>
      <th>343</th>
      <td>54</td>
      <td>100</td>
      <td>91</td>
      <td>62</td>
      <td>105</td>
      <td>197</td>
    </tr>
    <tr>
      <th>344</th>
      <td>4</td>
      <td>166</td>
      <td>193</td>
      <td>168</td>
      <td>2</td>
      <td>182</td>
    </tr>
    <tr>
      <th>345</th>
      <td>7</td>
      <td>85</td>
      <td>171</td>
      <td>60</td>
      <td>170</td>
      <td>131</td>
    </tr>
    <tr>
      <th>346</th>
      <td>79</td>
      <td>76</td>
      <td>45</td>
      <td>91</td>
      <td>23</td>
      <td>193</td>
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
      <td>170</td>
      <td>4</td>
      <td>4</td>
      <td>38</td>
      <td>177</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>197</td>
      <td>96</td>
      <td>2</td>
      <td>89</td>
      <td>119</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2</td>
      <td>12</td>
      <td>80</td>
      <td>46</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>60</td>
      <td>5</td>
      <td>5</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>28</td>
      <td>92</td>
      <td>26</td>
      <td>103</td>
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
      <td>6</td>
      <td>182</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>343</th>
      <td>2</td>
      <td>93</td>
      <td>61</td>
      <td>165</td>
      <td>75</td>
      <td>46</td>
    </tr>
    <tr>
      <th>344</th>
      <td>74</td>
      <td>52</td>
      <td>8</td>
      <td>3</td>
      <td>61</td>
      <td>153</td>
    </tr>
    <tr>
      <th>345</th>
      <td>8</td>
      <td>15</td>
      <td>91</td>
      <td>104</td>
      <td>91</td>
      <td>103</td>
    </tr>
    <tr>
      <th>346</th>
      <td>1</td>
      <td>72</td>
      <td>2</td>
      <td>2</td>
      <td>164</td>
      <td>93</td>
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
      <td>69</td>
      <td>21</td>
      <td>89</td>
      <td>123</td>
      <td>123</td>
      <td>89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>25</td>
      <td>65</td>
      <td>85</td>
      <td>89</td>
      <td>89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>38</td>
      <td>32</td>
      <td>36</td>
      <td>130</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>57</td>
      <td>25</td>
      <td>4</td>
      <td>160</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>83</td>
      <td>16</td>
      <td>112</td>
      <td>79</td>
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
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>34</td>
      <td>4</td>
    </tr>
    <tr>
      <th>343</th>
      <td>70</td>
      <td>2</td>
      <td>8</td>
      <td>62</td>
      <td>20</td>
      <td>85</td>
    </tr>
    <tr>
      <th>344</th>
      <td>1</td>
      <td>6</td>
      <td>110</td>
      <td>2</td>
      <td>2</td>
      <td>123</td>
    </tr>
    <tr>
      <th>345</th>
      <td>2</td>
      <td>2</td>
      <td>169</td>
      <td>38</td>
      <td>169</td>
      <td>104</td>
    </tr>
    <tr>
      <th>346</th>
      <td>3</td>
      <td>62</td>
      <td>2</td>
      <td>85</td>
      <td>73</td>
      <td>164</td>
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


    
![png](encoded_size_files/encoded_size_85_0.png)
    


#### Description of the results

##### Average results


```python
np.mean(BC_ratios)
```




    1.5021671797070961




```python
np.mean(MR_ratios)
```




    1.53880909133971




```python
np.mean(IaL_ratios)
```




    1.388001942116385




```python
np.mean(DI_ratios)
```




    1.3438366218935607



##### Median results


```python
np.median(BC_ratios)
```




    1.3719298223952499




```python
np.median(MR_ratios)
```




    1.4227232356937856




```python
np.median(IaL_ratios)
```




    1.197246197867642




```python
np.median(DI_ratios)
```




    1.1698512699354342



##### IQR


```python
def iqr(distrib):
    return np.percentile(distrib, 75) - np.percentile(distrib, 25)
```


```python
iqr(MR_ratios)
```




    0.5194317688380488




```python
iqr(BC_ratios)
```




    0.3609898631118664




```python
iqr(IaL_ratios)
```




    0.41750831584819226




```python
iqr(DI_ratios)
```




    0.41037543785989783



#### Statistical tests - Welch t-test


```python
stats.wilcoxon(IaL_ratios, BC_ratios)
```




    WilcoxonResult(statistic=16750.0, pvalue=6.640581673235631e-13)




```python
stats.wilcoxon(IaL_ratios, MR_ratios)
```




    WilcoxonResult(statistic=15949.0, pvalue=2.5753354000323727e-12)




```python
stats.wilcoxon(DI_ratios, BC_ratios)
```




    WilcoxonResult(statistic=15229.0, pvalue=1.2441054305481537e-15)




```python
stats.wilcoxon(DI_ratios, MR_ratios)
```




    WilcoxonResult(statistic=14650.0, pvalue=2.862552326287399e-14)



DI and MR, DI and BC, IaL and MR, IaL and BC are significantly different


```python
stats.wilcoxon(DI_ratios, IaL_ratios)
```




    WilcoxonResult(statistic=22367.0, pvalue=0.0033838096175471564)



IaL and DI are significantly different


```python
stats.wilcoxon(BC_ratios, MR_ratios)
```




    WilcoxonResult(statistic=22764.0, pvalue=7.16941159329483e-05)



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
      <td>1.315401</td>
      <td>05</td>
      <td>Beetle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.055879</td>
      <td>05</td>
      <td>Model Shift (MS)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.184716</td>
      <td>05</td>
      <td>No Transfer (NT)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.516075</td>
      <td>05</td>
      <td>Learning to Sample (L2S)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.512912</td>
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
      <td>1.022233</td>
      <td>25</td>
      <td>Learning to Sample (L2S)</td>
    </tr>
    <tr>
      <th>8324</th>
      <td>1.260107</td>
      <td>30</td>
      <td>Beetle</td>
    </tr>
    <tr>
      <th>8325</th>
      <td>1.310221</td>
      <td>30</td>
      <td>Model Shift (MS)</td>
    </tr>
    <tr>
      <th>8326</th>
      <td>1.003280</td>
      <td>30</td>
      <td>No Transfer (NT)</td>
    </tr>
    <tr>
      <th>8327</th>
      <td>1.001359</td>
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
plt.ylim(0.9,2.5)
plt.legend(fontsize=20, loc = 'upper right', framealpha=1)


# Arrow
plt.arrow(x = 4.5, y = 2, dx= 0, dy = -0.3, head_width = 0.15, head_length = .1, color="orange")
plt.text(s = "Better", x = 4, y = 1.95, size = 20, color = 'orange')

plt.xticks(size = 20)
plt.yticks(size = 20)
plt.savefig("../../results/res_box_tl_approach.png")
plt.show()
```


    
![png](encoded_size_files/encoded_size_116_0.png)
    



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




    1.4301873762071775




```python
np.median(get_ratio("Model Shift (MS)", 10))
```




    1.2575729493731396




```python
iqr(get_ratio("Model Shift (MS)", 10))
```




    0.3458383233351794




```python
np.mean(get_ratio("Beetle", 10))
```




    1.3129732513534162




```python
np.median(get_ratio("Beetle", 10))
```




    1.1993260625307542




```python
iqr(get_ratio("Beetle", 10))
```




    0.29982866711636147




```python
np.median(get_ratio("Learning to Sample (L2S)", 10))
```




    1.1459765936661719




```python
stats.wilcoxon(DI_ratios, get_ratio("Learning to Sample (L2S)", 30))
```




    WilcoxonResult(statistic=8223.0, pvalue=8.174376176373985e-29)




```python
np.mean(get_ratio("Learning to Sample (L2S)", 30))
```




    1.0847088518502517




```python
np.median(get_ratio("Learning to Sample (L2S)", 30))
```




    1.0507705752630465




```python
np.mean(get_ratio("Learning to Sample (L2S)", 30)==1)
```




    0.12968299711815562




```python
iqr(get_ratio("Learning to Sample (L2S)", 10))
```




    0.2020173487258652




```python
iqr(get_ratio("Learning to Sample (L2S)", 30))
```




    0.09721825002503048




```python
iqr(get_ratio("Beetle", 5))
```




    0.3150641863954504




```python
iqr(get_ratio("Beetle", 30))
```




    0.2812094844386206




```python
iqr(get_ratio("Model Shift (MS)", 10))
```




    0.3458383233351794




```python
iqr(get_ratio("Model Shift (MS)", 30))
```




    0.2817078922777032




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
