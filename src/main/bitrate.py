#!/usr/bin/env python
# coding: utf-8

# # Learning Once and for All - On the Input Sensitivity of Configurable Systems
# 
# ### This notebook details the main results presented in the paper submitted to the International Conference of Software Engineering.

# #### Warning; Before launching the notebook, make sure you have installed all the packages in your python environment
# #### To do that,  open a terminal in the replication folder, and use the requirements.txt file to download the libraries needed for this script :
# `pip3 install -r requirements.txt`
# #### If it worked, you should be able to launch the following cell to import libraries.

# In[1]:

print("Import libraries...")

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

# we use it to interact with the file system
import os
# compute time
from time import time

print("Done!")

# #### Now, we import data

# In[2]:

print("Import measurements...")

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

print("Done!")

# In[3]:


print("We consider ", len(listVideo), " videos")


# #### Just a few words about the time needed to compute the data; about a month is needed to fully replicate the experience

# In[4]:


totalTime = np.sum([np.sum(vid["etime"]) for vid in listVideo])
#print("Hours of computation needed : "+str(totalTime/(3600)))
print("Days of computation of measurement: "+str(totalTime/(24*3600)))


# #### Our focus in this paper is the bitrate, in kilobits per second

# In[5]:

#our variable of interest
predDimension = "kbs"

print("Performance considered :",predDimension)


for i in range(len(listVideo)):
    sizes = listVideo[i][predDimension]
    ind = sorted(range(len(sizes)), key=lambda k: sizes[k])
    listVideo[i]['ranking'] = ind


# # In the paper, here starts Section II

# # RQ1 - Do Input Videos Change Performances of x264 Configurations?

# ## RQ1.1 - Do software performances stay consistent across inputs?

# #### A-] For this research question, we computed a matrix of Spearman correlations between each pair of input videos.
# 
# Other alternatives : Kullback-Leibler divergences to detect outliers, and Pearson correlation to compute only linear correlations 

print("Section II in the paper")

print("RQ1 - Do Input Videos Change Performances of x264 Configurations?")

print("RQ1.1 - Do software performances stay consistent across inputs?")

# In[6]:

print("Computation of Spearman correlations... Let's start the coffee machine, it takes a while :-)")

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


# #### Here is the distribution depicted in figure 1, on the bottom left; we removed the diagonal $i!=j$ in the following code.

# In[7]:


corrDescription = [corrSpearman[i][j] for i in range(nbVideos) for j in range(nbVideos) if i >j]
pd.Series(corrDescription).describe()

print("Done!")

# #### Few statistics about input videos, mentioned in the text

# #### A small detail; in the paper, when we mention the video having the id $i$,  it means the $(i+1)^{th}$ video of the list, because the first input has the index 0

# In[8]:


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


# In[9]:


corrSpearman[378][1192]


# In[10]:


corrSpearman[314][1192]


# In[11]:


corrSpearman[378][314]


# #### "For 95% of the videos, it is always possible to find another video having a correlation higher than 0.92" -> here is the proof 

# In[12]:


argm = [np.max([k for k in corrSpearman[i] if k <1]) for i in range(len(corrSpearman))]
pd.Series(argm).describe()


# In[13]:


np.percentile(argm, 5)


# ## Figure 1
# 
# #### Now, let's compute figure 1!

# In[14]:


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

print("Computation of Performance groups... Enjoy your coffee, it will last few minutes!")

group_no_ordered = plot_correlationmatrix_dendogram(corrSpearman, 
                                 "corrmatrix-ugc-dendo-Spearman-" + predDimension + ".pdf",
                                 [k/5 for k in np.arange(-10,10,1)], method='ward')

print("Done!")

# #### To match the increasing number of groups to the order of the figure (from the left to the right), we change the ids of groups

# In[15]:


map_group = [2, 0, 3, 1]

def f(gr):
    return map_group[int(gr)]

# we apply this mapping
groups = np.array([*map(f, group_no_ordered)],int)

print("Group 1 contains", sum(groups==0), "input videos.")
print("Group 2 contains", sum(groups==1), "input videos.")
print("Group 3 contains", sum(groups==2), "input videos.")
print("Group 4 contains", sum(groups==3), "input videos.")


# ### B-] We also study rankings of configurations

# #### First, we compute the rankings

# In[16]:

print("Rankings of configurations")

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


# #### To get the most "unstable" ranking, we take the configuration having the highest standard deviation.

# In[17]:


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
plt.savefig("../../results/config_min_std_ranking.png")
plt.show()

# depicts the most input sensitive configuration
plt.scatter(range(len(rankings.columns)), rankings.loc[np.argmax(stds), :])
plt.ylabel("Rank")
plt.xlabel("Video id")
plt.title("Configuration "+str(np.argmax(stds))+" : std = "+str(int(np.round(np.max(stds)))))
plt.savefig("../../results/config_max_std_ranking.png")
plt.show()


# #### Some statistics (not mentioned in the text)

# In[18]:


print("For config. 200, ", int(np.sum([1  for j in range(len(rankings.loc[np.argmin(stds),:])) 
              if rankings.loc[np.argmin(stds),:][j] > 105 and rankings.loc[np.argmin(stds),:][j] < 130])
      /len(rankings.loc[np.argmin(stds),:])*100),"% of configurations are between 105 and 130!")


# In[19]:


np.where(rankings.loc[np.argmin(stds),:] == np.min(rankings.loc[np.argmin(stds),:]))


# In[20]:


np.max(rankings.loc[np.argmax(stds),:])


# In[21]:


np.where(rankings.loc[np.argmax(stds),:] == np.min(rankings.loc[np.argmax(stds),:]))


# In[22]:


np.max(rankings.loc[np.argmax(stds),:])


# In[23]:


np.where(rankings.loc[np.argmin(stds),:] == np.max(rankings.loc[np.argmin(stds),:]))


# #### Rankings distributions

# In[24]:


pd.Series(rankings.loc[np.argmax(stds),:]).describe()


# In[25]:


pd.Series(rankings.loc[np.argmin(stds),:]).describe()

print("config... Done!")

# ## RQ1-2- Are there some configuration options more sensitive to input videos?

# #### A-] For RQ1-2, we compute the feature importances of configuration options for each video

# In[26]:

print("RQ1-2- Are there some configuration options more sensitive to input videos?")

print("Computation of feature importances - You can check our repo while the code works for you, or read our paper if you're interested!")

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


# In[27]:


# we compute the feature importances
res_imp = compute_Importances(listVideo)

print("Computation of feature importances - Done!")

# ## Figure 2a
# #### Then, we depict a boxplot of features importances; for each feature, there are 1397 feature importances (one per video)

# In[28]:


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

print("Figure 2a generated")

# #### B-] Since feature importances do not get how the predicting variables (i.e. the configuraiton options) affect the variable to predict (i.e. the bitrate), we add linear regression coefficients

# In[29]:

print("Computation of feature effects - Start! Btw, did you check the information subfolder (in replication/information)? It details our working environement!")

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

print("Computation of feature effects - Done!")

# ## Figure 2b
# #### Same idea for this plot, see the last cell of RQ1.2-A-]

# In[30]:


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


print("Figure 2b, done!")

# # In the paper, here starts Section III

# ## RQ1bis - Can we group together videos having same performance distributions?

# #### We use figure 1 groups to isolate encoding profile of input videos associated to bitrate

# ### We load the metrics of the youtube UGC dataset, needed for RQ2

# In[31]:

print("In the paper, here starts Section III")

print("Load the videos metadata!")

# we load the file (in itself an aggregation of datasets)
# the file is available in the data folder, then ugc_meta
# each line is a video, and the columns are the different metrics
# provided by Wang et. al.
meta = pd.read_csv("../../data/ugc/ugc_meta/all_features.csv").set_index('FILENAME')
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
perf = pd.DataFrame({'FILENAME': np.array([v_names[k][:-4] for k in range(len(v_names))]),
              'perf_group' : np.array([k for k in groups])}).set_index('FILENAME')
meta_perf = perf.join(meta)
# print the results
meta_perf


# #### We compute the count of categories per group

# In[32]:


# keep str categories, to detect which categories are more represented per performance group
meta_perf['str_video_cat'] = [str(meta_perf.index[i]).split('_')[0] for i in range(meta_perf.shape[0])]
# count the occurence
total_cat = meta_perf.groupby('str_video_cat').count()['perf_group']
# performance group per input id
group_perf = np.array([gr for gr in groups])
group_perf


# #### We define a function to depict a boxplot

# In[33]:


def boxplot_imp(res, xlim = None, criteria = 'max', name = None, xname='Feature importances'):
    # sort features by decreasing Q3 (max up, min down)
    if criteria == 'max':
        listImp = [(np.percentile(res[col],75), res[col], col) 
                   for col in res.columns]
        listImp.sort(key=lambda tup: tup[0])
    # sort features by decreasing Inter Quartile Range (max up, min down)
    elif criteria == 'range':
        listImp = [(np.abs(np.percentile(res[col],75)-np.percentile(res[col],25)),res[col], col) 
                   for col in res.columns]
        listImp.sort(key=lambda tup: tup[0])
    # sort features by names, (A up, Z down)
    elif criteria == 'name':
        listImp = [(np.abs(np.percentile(res[col],75)-np.percentile(res[col],25)),res[col], col) 
                   for col in res.columns]
        listImp.sort(key=lambda tup: tup[2], reverse=True)
    
    # see figures 2a and 2b
    red_square = dict(markerfacecolor='r', marker='s')
    plt.figure(figsize=(15,8))
    plt.grid()
    plt.boxplot([l[1] for l in listImp], flierprops=red_square, 
              vert=False, patch_artist=True, #widths=0.25,
              boxprops=dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
              whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))
    plt.xlabel(xname, size=13)
    if xlim:
        plt.xlim(xlim)
    plt.yticks(range(1, len(listImp) + 1), [l[2] for l in listImp])
    if name:
        plt.savefig(name)
    plt.show()


# In[34]:


input_sizes = pd.read_csv("../../data/ugc/ugc_meta/sizes.csv", delimiter=',').set_index('name')

print("Finished!")
# ## Figure 4
# 
# #### Summary for each group
# 
# #### The idea of this part is to show that groups of performance are homogeneous; in a group, the inputs have a lot in common
# #### Ok, same bitrates, but that's just the obvious part !!! 
# #### Other raw performances, feature importances, linear reg coefficients, they are 
# #### These groups are semantically valids and extend the classification of 2004, established by Maxiaguine et al.
# 
# Interestingly, groups formed by encoded sizes (with respect to the same protocol gives the same groups, except for 30 videos (going from the fourth to the third group)

# In[35]:

print("Group descriptions... start!")

# [begin what if] they should be already defined, to remove?
listFeatures = ["cabac", "ref", "deblock", "analyse", "me", "subme", "mixed_ref", "me_range", 
                "trellis", "8x8dct", "fast_pskip", "chroma_qp_offset", "bframes", "b_pyramid", 
                "b_adapt", "direct", "weightb", "open_gop", "weightp", "scenecut", "rc_lookahead", 
                "mbtree", "qpmax", "aq-mode"]

to_keep = [k for k in listFeatures]
to_keep.append(predDimension)

categorial = ['analyse', 'me', 'direct']
# [end what if]

# computes a summary 
def summary_group(id_group):
    # input : A group id, see Figure 1 of the paper, from 0 to 3 (1 to 4 in the paper)
    # output: in the console (not optimized yet to output directly the table)
    
    # first, we isolate all the ids of one group
    id_list = [i for i in range(len(listVideo)) if group_perf[i]==id_group]
    v_names_group = [v_names[i][:-4] for i in range(len(v_names)) if i in id_list]
    listVideoGroup = [listVideo[i] for i in range(len(listVideo)) if i in id_list]
    
    # print the ids, plus the number of videos
    print('Group '+str(id_group)+' : '+str(len(listVideoGroup))+' videos!')
    
    print('\n')
    
    # average sizes of videos (Warning : BEFORE COMPRESSION!) per group
    video_size = [input_sizes.loc[index_vid]['size'] for index_vid in v_names_group]
    
    # mean and stds
    print("input avg size :", np.mean(video_size))
    print("input std size :", np.std(video_size))
    
    print('\n')

    # features importances per groupfor influential features
    res_imp = compute_Importances(listVideoGroup)
    
    print('\n')
    
    # for mbtree
    print('Imp mbtree:', np.mean(res_imp['mbtree']))
    print('Imp std mbtree:', np.std(res_imp['mbtree']))
    
    # for aq-mode
    print('Imp aq-mode:', np.mean(res_imp['aq-mode']))
    print('Imp std aq-mode:', np.std(res_imp['aq-mode']))
    
    # for subme
    print('Imp subme:', np.mean(res_imp['subme']))
    print('Imp std subme:', np.std(res_imp['subme']))
    
    # see the previous cell, boxplot of feature importances per group
    boxplot_imp(res_imp, criteria = 'name', xlim= (0, 1),
               name ="../../results/boxplot_imp_group"+str(id_group)+".png")

    # features effects
    res_poly = compute_poly(listVideoGroup)
    
    print('\n')
    
    # mean and stds for influential features, same as with rf feature importances
    print('Coef mbtree:', np.mean(res_poly['mbtree']))
    print('Coef mbtree std :', np.std(res_poly['mbtree']))
    print('Coef aq-mode:', np.mean(res_poly['aq-mode']))
    print('Coef aq_mode std :', np.std(res_poly['aq-mode']))
    print('Coef subme:', np.mean(res_poly['subme']))
    print('Coef subme std:', np.std(res_poly['subme']))
    
    # Boxplot of linear coefficients
    boxplot_imp(res_poly, criteria = 'name', xlim = (-1, 1),
               name ="../../results/boxplot_effect_group"+str(id_group)+".png", xname = 'Coefficients')

    print('\n')
    
    # The five performances we measured during this experience
    interest_var = ['cpu', 'etime', 'fps', 'kbs', 'size']
    
    # mean and stds
    for iv in interest_var:
        pred = [np.mean(lv[iv]) for lv in listVideoGroup]
        print('Mean '+iv+' in the group: '+str(np.round(np.mean(pred),1)))
        print('Std dev : '+iv+' in the group: '+str(np.round(np.std(pred),1)))

    print('\n')

    # percentage of the videos present in the group per category
    meta_perf_group = meta_perf.query('perf_group=='+str(id_group))
    meta_perf_group['str_video_cat'] = [str(meta_perf_group.index[i]).split('_')[0] for i in range(meta_perf_group.shape[0])]
    val_group = meta_perf_group.groupby('str_video_cat').count()['perf_group']
    df_res_cat_group = pd.DataFrame({'val': val_group, 'total': total_cat})
    print(df_res_cat_group['val']/df_res_cat_group['total'])

    print('\n')

    # Mean of the videos of the group per properties
    for col in meta_perf_group.columns:
        if col not in ['str_video_cat', 'video_category']:
            print('Mean '+col+' : '+str(meta_perf_group[col].mean()))
            print('std '+col+' : '+str(meta_perf_group[col].std()))

    print('\n')
    
    # Spearman Correlations intra-groups
    corrGroup = np.array([corrSpearman[i][j] for i in range(len(corrSpearman)) if i in id_list 
                 for j in range(len(corrSpearman)) if j in id_list],float)

    print("Correlations intra-group: \n" + str(pd.Series(corrGroup).describe())+'\n')


# In[36]:

print("Group 1")
summary_group(0)


# In[37]:

print("Group 2")
summary_group(1)


# In[38]:

print("Group 3")
summary_group(2)


# In[39]:

print("Group 4")
summary_group(3)


print("Figure 4 is an aggregation of the previous outputs.")
# ### Inter-group correlogram

# In[40]:

print("Inter-group correlations... start!")
group_perf =groups
id_list_0 = [i for i in range(len(listVideo)) if group_perf[i]==1]
id_list_1 = [i for i in range(len(listVideo)) if group_perf[i]==3]
id_list_2 = [i for i in range(len(listVideo)) if group_perf[i]==0]
id_list_3 = [i for i in range(len(listVideo)) if group_perf[i]==2]

res = np.zeros(16).reshape(4,4)
stds = np.zeros(16).reshape(4,4)
res_med = np.zeros(16).reshape(4,4)

tab = []
for id0 in id_list_0:
    for id1 in id_list_1:
        tab.append(corrSpearman[id0][id1])
res[0][1] = np.mean(tab)
stds[0][1] = np.std(tab)
res_med[0][1] = np.median(tab)

tab = []
for id0 in id_list_0:
    for id2 in id_list_2:
        tab.append(corrSpearman[id0][id2])
res[0][2] = np.mean(tab)
stds[0][2] = np.std(tab)
res_med[0][2] = np.median(tab)

tab = []
for id0 in id_list_0:
    for id3 in id_list_3:
        tab.append(corrSpearman[id0][id3])
res[0][3] = np.mean(tab)
stds[0][3] = np.std(tab)
res_med[0][3] = np.median(tab)

tab = []
for id1 in id_list_1:
    for id2 in id_list_2:
        tab.append(corrSpearman[id1][id2])
res[1][2] = np.mean(tab)
stds[1][2] = np.std(tab)
res_med[1][2] = np.median(tab)

tab = []
for id1 in id_list_1:
    for id3 in id_list_3:
        tab.append(corrSpearman[id1][id3])
res[1][3] = np.mean(tab)
stds[1][3] = np.std(tab)
res_med[1][3] = np.median(tab)

tab = []
for id2 in id_list_2:
    for id3 in id_list_3:
        tab.append(corrSpearman[id2][id3])
res[2][3] = np.mean(tab)
stds[2][3] = np.std(tab)
res_med[2][3] = np.median(tab)


res[0][0] = np.mean([[corrSpearman[id1][id2] for id1 in id_list_0] for id2 in id_list_0])
res[1][1] = np.mean([[corrSpearman[id1][id2] for id1 in id_list_1] for id2 in id_list_1])
res[2][2] = np.mean([[corrSpearman[id1][id2] for id1 in id_list_2] for id2 in id_list_2])
res[3][3] = np.mean([[corrSpearman[id1][id2] for id1 in id_list_3] for id2 in id_list_3])

print("AVG")
print(res)
print("STD")
print(stds)
print('MEDIAN')
print(res_med)

print("Inter-group correlations... start!")
# #### In a group (correlation intra), the performances are highly or very highly correlated
# 
# #### Between the different (correlaiton inter), the performances of inputs are generally moderate or low (except for groups 3 and 4)

# ## INTUITION of Inputec : why should we use the metrics of the youtube UGC Dataset to discriminate the videos into groups?
# 
# Due to the lack of space, we didn't explain this experiment in the paper; but we still think it is an important milestone to understand how we've got the idea of Inputec!
# 
# ### We used the metrics of Youtube UGC to classify each video in its performance group.
# 
# ### RESULTS : in average, we classify successfully two videos over three (~66%) in the right performance group

# In[41]:


if 'str_video_cat' in meta_perf.columns:
    del meta_perf['str_video_cat']

accuracy = []

nbLaunches = 10
for i in range(nbLaunches):
    X = np.array(meta_perf[[k for k in meta_perf.columns if k !='perf_group']], float)
    y = np.array(meta_perf['perf_group'], float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)


    conf = pd.crosstab(y_pred, y_test)#, colnames=[1,2,3], rownames=[1,2,3])
    val = np.sum(np.diag(conf))/len(y_test)
    accuracy.append(val)
    print('Test accuracy : '+ str(val))
    conf.columns = pd.Int64Index([1,2,3,4], dtype='int64', name='Observed')
    conf.index = pd.Int64Index([1,2,3,4], dtype='int64', name='Predicted')
    conf
print(np.mean(accuracy))
conf


# # In the paper, here starts Section IV

# # RQ2 - Can we use Inputec to find configurations adapted to input videos?

# ### The goal fixed in RQ2 is to generate a configuration minimizing the bitrate for a given video!

# In[42]:

print("Section IV in ther paper")

print("Initialisation")

listFeatures = ["cabac", "ref", "deblock", "analyse", "me", "subme", "mixed_ref", "me_range", "trellis", 
                "8x8dct", "fast_pskip", "chroma_qp_offset", "bframes", "b_pyramid", 
                "b_adapt", "direct", "weightb", "open_gop", "weightp", "scenecut", "rc_lookahead", 
                "mbtree", "qpmax", "aq-mode"]
categorial = ['analyse', 'me', 'direct']
val_config = listVideo[0][listFeatures].replace(to_replace ="None",value='0')
val_config['deblock'] =[int(val[0]) for val in val_config['deblock']]

for col in val_config.columns:
    if col not in categorial:
        arr_col = np.array(val_config[col],int)
        arr_col = (arr_col-np.mean(arr_col))/(np.std(arr_col)+1e-5)
        val_config[col] = arr_col
    else:
        if col not in [predDimension,'ranking']:
            val_config[col] = [np.where(k==val_config[col].unique())[0][0] for k in val_config[col]]
            arr_col = np.array(val_config[col],int)
            arr_col = (arr_col-np.mean(arr_col))/(np.std(arr_col)+1e-5)
            val_config[col] = arr_col


# #### Function to "place" a value (i.e. give a rank to a value) in an ordered list

# In[43]:


def find_rank(sorted_perfs, val):
    # inputs : a list of sorted performances, a value 
    # output: the ranking of value in the sorted_perf list 
    rank = 0
    while val > sorted_perfs[rank] and rank < len(sorted_perfs)-1:
        rank+=1
    return rank


# ## Method M1 - Inputec, full configurations, full properties
# 
# ## [1 sentence explanation] We included the properties in the set of predicting variables
# 
# #### [short explanation] Offline: by including the input properties, we discriminate the input videos into performance groups, thus increasing the mean absolute error of the prediction. Online: instead of measuring new configurations (as known as transfer learning), we compute the input properties, and test all the configurations. At the end, we select the one giving the minimal prediction.

# ## OFFLINE
# 
# #### [OFFLINE] Construct the data
# 
# Lines 1-12 in Algorithm 1

# In[44]:

print("Inputec training : it takes a while, again... Sorry for the inconvenience! ")

# we separate the list of videos into a training (i.e. offline) set and a test set (i.e. online)
train_ind, test_ind = train_test_split([k for k in range(len(listVideo))], test_size = 0.25)
# training set indexes
train_index = [v_names[k][:-4] for k in train_ind]

# we add the input properties
name_col = list(meta_perf.columns)[1:]

# we add the x264 configuration options
for vcc in val_config.columns:
    name_col.append(vcc)

# Then, X (i.e the predicting variables) =  input properties + software configuration options

# we add the variable to predict, i.e. y the bitrate performance distribution
name_col.append("bitrate")

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
    # LINES 6-10 in Algorithm 1
    for i in range(len(inputs_names)):
        # first, we retrieve the name of the video
        video_name = inputs_names[i]
        index_video = np.where(np.array([v[:-4] for v in v_names], str)==video_name)[0][0]
        # we compute the performance, here
        bitrates = listVideo[index_video][predDimension]
        # get the input properties of the video
        video_prop = np.array(meta_perf.loc[video_name][1:], float)
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
X_train = np.array(training_data.drop(["bitrate"],axis=1), float)
y_train = np.array(training_data["bitrate"], float)


# #### [OFFLINE] We train the Learning Algorithm
# 
# Lines 13-14 in Algorithm 1

# In[45]:


# The hyperparameters were optimized by testing different values for parameters
# and comparing the mean absolute error given by the model
LA = RandomForestRegressor(n_estimators=100, criterion="mse", min_samples_leaf=2, bootstrap=True, 
                           max_depth=None, max_features=15)
# the default config for random forest is quite good
# we train the model LA
LA.fit(X_train, y_train)


# ## ONLINE

# #### [ONLINE] Add new videos
# 
# Lines 15-17 in the Algorithm 1

# In[46]:


# test set indexes
test_index = [v_names[k][:-4] for k in test_ind]


# #### [ONLINE] Predict the value for each configuration, and output the configuration giving the minimal result (i.e. the argmin)
# 
# Lines 18-22 in the Algorithm 1

# In[47]:


# we compute the time - start
start = time()

# the performance values for the configuration chosen by Inputec, for the test set of videos
inputec = []

# the performance rankings for the configuration chosen by Inputec, for the test set of videos
inputec_ranks = []

# for each video in the test set
for ti in test_index:
    # we retrieve the test index
    index_video = np.where(np.array([v[:-4] for v in v_names], str)==ti)[0][0]
    # list of predictions for inputec
    pred_inputec = []
    # for each configuration
    for i in range(nb_config):
        # we add input properties to the configurations
        video_prop = list(tuple(meta_perf.loc[ti][1:])+tuple(val_config.loc[i]))
        # we predict the value associated to the configuration
        pred_inputec.append(LA.predict(np.array(video_prop, float).reshape(1, 33)))
    # sorted performances
    sorted_perfs = sorted(listVideo[index_video][predDimension])
    # the performance pof the configuration chosen by Inputec
    perf = listVideo[index_video].loc[np.argmin(pred_inputec)][predDimension]
    # we add it to the perf array
    inputec.append(perf)
    # the related ranking (between 0 and 200, hopefully close to 0)
    inputec_ranks.append(find_rank(sorted_perfs, perf))

# we compute the time - end
end = time()

print("Average time for one video prediction < ", int(100*(end-start)/len(test_index))/100, "second(s)!")

print("Inputec training... Done! ")
print("Baseline definitions... Start!")
# ## Baselines
# 
# To evaluate Inputec, we compare it to different baselines. 
# Each baseline corresponds to a concrete situation:

# #### B1 - Model reuse
# 
# We arbitrarily choose a first video, learn a performance model on it, and select the best configuration minimizing the performance for this video. This baseline represents the error made by a model trained on a  source input (i.e., a  first video) and transposed to a target input (i.e., a second video, different from the first one), without considering the difference of content between the source and the target. In theory, it corresponds to a fixed configuration, optimized for the first video. We add B1 to measure how we can improve the standard performance model with Inputec.
# 
# B1 is fixed.

# In[55]:


# we arbitraly select an input in the training set
# here we choose on purpose a video for which the model reuse leads to bad results
# to see what could happen when we just reuse the model from one video to another
chosen_input = 423

# we consider the associated input video
source_video = listVideo[np.where(np.array([v[:-4] for v in v_names], str)==
                                  train_index[chosen_input])[0][0]]

# we select the best config for this video
b1_config = np.argmin(source_video[predDimension])

print("Id of B1 configuration :", b1_config)

# the rankings of the first baseline
b1_ranks = []

# for each video in the test set
for ti in test_index:
    # we retrieve the test index of the input video
    index_video = np.where(np.array([v[:-4] for v in v_names], str)==ti)[0][0]
    # sorted performances
    sorted_perfs = sorted(listVideo[index_video][predDimension])
    # the performance of the software configuration of B1
    perf = listVideo[index_video].loc[b1_config][predDimension]
    # we add it to the list
    b1_ranks.append(find_rank(sorted_perfs, perf))


# #### B2 - Best compromise
# 
# We select the configuration having the lowest sum of bitrates rankings for the training set of videos, and study this configuration's distribution on the validation set. 
# B2 represents the best compromise we can find, working for all input videos. 
# In terms of software engineering, it acts like a preset configuration proposed by x264 developers.
# Beating this configuration shows that our approach chooses a custom configuration, tailored for the input characteristics.
# 
# B2 is fixed.

# In[56]:


# only keep the video of the training set (we keep the training-test phases)
keep_vid = ['video'+str(np.where(np.array([v[:-4] for v in v_names], str)==ti)[0][0]) for ti in train_index]

# select the best compromise, i.e. the configuration having the minimal sum of rankings
b2_config = np.argmin([np.sum(np.array(rankings[keep_vid].loc[i], int)) 
                             for i in range(rankings.shape[0])])

print("Id of B2 configuration :", b2_config)

# the rankings of the second baseline
b2_ranks = []

# for each video in the test set
for ti in test_index:
    # we retrieve the test index of the input video
    index_video = np.where(np.array([v[:-4] for v in v_names], str)==ti)[0][0]
    # sorted performances
    sorted_perfs = sorted(listVideo[index_video][predDimension])
    # the performance of the software configuration of B1
    perf = listVideo[index_video].loc[b2_config][predDimension]
    # we add it to the list
    b2_ranks.append(find_rank(sorted_perfs, perf))


# #### B3 - Average performance
# 
# This baseline computes the average performance of configurations for each video of the validation dataset. 
# It acts as a witness group, reproducing the behavior of a non-expert user that experiments x264 for the first time, and selects uniformly one of the 201 configurations of our dataset.
# 
# B3 vary across inputs.

# In[57]:


# the average performance values for all configurations, for each input in the test set of videos
b3_perf = []

# the rankings of the third baseline
b3_ranks = []

# for each video in the test set
for ti in test_index:
    # we retrieve the test index
    index_video = np.where(np.array([v[:-4] for v in v_names], str)==ti)[0][0]
    # sorted performances
    sorted_perfs = sorted(listVideo[index_video][predDimension])
    # the average performance of the video
    perf = np.mean(listVideo[index_video][predDimension])
    # we add it to B3's performance array 
    b3_perf.append(perf)
    # we add it to the list
    b3_ranks.append(find_rank(sorted_perfs, perf))


# #### B4 - Best configuration
# 
# Similarly, we select the best configuration (i.e. leading to the minimal performance for the set of configurations). We consider this configuration as the upper limit of the potential gain of performance; since our approach chooses a configuration in the set of 201 possible choices, we can not beat the best one of the set; it just shows how far we are from the best performance value. 
# Otherwise, either our method is not efficient enough to capture the characteristics of each video, or the input sensitivity does not represent a problem, showing that we can use an appropriate but fixed configuration to optimize the performances for all input videos.
# 
# B4 vary across inputs.

# In[58]:


# the minimal performance values, for each input in the test set of videos
b4_perf = []

# the minimal performance values rankings
b4_ranks = np.zeros(len(test_index))

# for each video in the test set
for ti in test_index:
    # we retrieve the test index
    index_video = np.where(np.array([v[:-4] for v in v_names], str)==ti)[0][0]
    # we add the minimal performance of the video
    b4_perf.append(np.min(listVideo[index_video][predDimension]))


# ### Ratios
# 
# The ratios of Inputec over the baseline performances prediction should be lower than 1 if and only if Inputec is better than the baseline (because it provides a lower bitrate than the baseline).
# As an example, for a ratio of $0.6 = 1 - 0.4 = 1 - \frac{40}{100}$, we gain 40% of bitrate with our method compared to the baseline. 
# Oppositely, we loose 7% of bitrate with our method compared to the baseline for a ratio of 1.07.

# In[60]:


# We compute the four ratios

# Inputec/B1, the model reuse
ratio_b1 = []
# Inputec/B2, the compromise
ratio_b2 = []
# Inputec/B3, the average bitrate
ratio_b3 = []
# Inputec/B4, the best configuration
ratio_b4 = []

# for each video, we add the ratio to the list
for i in range(len(test_index)):
    index_video = np.where(np.array([v[:-4] for v in v_names], str)==test_index[i])[0][0]
    
    # for B1 and B2, we take the configuration of the current video
    ratio_b1.append(inputec[i]/listVideo[index_video].loc[b1_config][predDimension])
    ratio_b2.append(inputec[i]/listVideo[index_video].loc[b2_config][predDimension])
    
    # for B3 and B4, we take the previously computed values
    ratio_b3.append(inputec[i]/b3_perf[i])
    ratio_b4.append(inputec[i]/b4_perf[i])


# ## Figure 5a - Performance ratios of configurations, baseline vs Inputec

# In[61]:


# we aggregate the different ratios, sorted by increasing efficiency
box_res = np.transpose(pd.DataFrame({"mean" : ratio_b3,
                                     "compromise" : ratio_b2,
                                     "model reuse" : ratio_b1,
                                     "min" : ratio_b4}))

# rotation of the text in the ordered axis, to fit the figure in the paper
degrees = 20

# cosmetic choices
red_square = dict(markerfacecolor='r', marker='s')
# figure size
plt.figure(figsize=(16,8))
# add a grid
plt.grid()
plt.boxplot(box_res, flierprops=red_square, 
          vert=False, patch_artist=True, #widths=0.25,
          boxprops=dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
          whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))
# add crosses for average values of distributions
plt.scatter(np.array([np.mean(box_res.iloc[i]) for i in range(4)]), np.arange(1, 5, 1), 
            marker="x", color = "red", alpha = 1, s = 100)
# Limits
plt.ylim(0,5)
plt.xlim(0,2)
# Inputec vs Baseline
plt.vlines(x=1, ymin=0.5, ymax=4.5, linestyle='-.', color='green', linewidth = 5)
plt.text(s = "Inputec worse than Baseline", x = 1.2, y = 0.3, size = 20, color = 'green')
plt.text(s = "Inputec better than Baseline", x = 0.2, y = 0.3, size = 20, color = 'green')
# Labels
plt.xlabel("Ratio Inputec/Baseline", size = 20)
# Ticks of axis
plt.yticks([1, 2, 3, 4], 
           ['B3 - Avg', 'B2 - Preset', 'B1 - Reuse', 'B4 - Best'],
          size = 20,
          rotation=degrees)
plt.xticks(size = 20)
# save figure in the results folder
plt.savefig("../../results/res_box_baseline.png")
# show the figure
plt.show()

print("Baseline definitions... Done!")
print("Figure 5a ok!")

# ## Figure 5b - Rankings of configurations, baseline vs Inputec

# In[62]:


box_res = np.transpose(pd.DataFrame({"mean" : b3_ranks,
                                     "compromise" : b2_ranks,
                                     "model reuse" : b1_ranks,
                                     "inputec" : inputec_ranks,
                                     "min" : b4_ranks}))
red_square = dict(markerfacecolor='r', marker='s')
plt.figure(figsize=(16,8))
plt.grid()
plt.boxplot(box_res, flierprops = red_square, 
          vert = False, patch_artist = True, #widths=0.25,
          boxprops = dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
          whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))
plt.scatter(np.array([np.mean(box_res.iloc[i]) for i in range(4)]), np.arange(1, 5, 1), 
            marker="x", color = "red", alpha = 1, s = 100)
plt.ylim(0, 6)
plt.xlim(-1,200)
plt.xlabel("Ranking of configurations", size = 20)
plt.arrow(x = 175, y = 3.5, dx= 0, dy = 1.5, head_width = 4, head_length = .15, color="orange")
plt.text(s = "Better", x = 180, y = 4.2, size = 20, color = 'orange')
plt.arrow(x = 154, y = 5.15, dx= -30, dy = 0, head_width = .15, head_length = 3, color="orange")
plt.text(s = "Better", x = 130, y = 4.8, size = 20, color = 'orange')
plt.yticks([1, 2, 3, 4, 5,], 
           ['B3 - Avg', 'B2 - Preset', 'B1 - Reuse', 'Inputec', 'B4 - Best'], 
           size=20,
          rotation=degrees)
plt.xticks(size=20)
plt.savefig("../../results/res_box_baseline_rank.png")
plt.show()

print("Figure 5b ok!")

# #### Statistical tests - Welch t-test

# In[63]:

print("Statistical tests - Welch t-tests")
print("Inputec vs b3")
print(stats.ttest_ind(inputec_ranks, b3_ranks, equal_var = False))


# In[64]:

print("Inputec vs b2")
print(stats.ttest_ind(inputec_ranks, b2_ranks, equal_var = False))


# In[65]:

print("Inputec vs b1")
print(stats.ttest_ind(inputec_ranks, b1_ranks, equal_var = False))


# Three Welch’s t-tests confirm that the rankings of Inputecare significantly different from B1, B2 and B3 rankings. 
# 
# We reject the null hypothesis (i.e. the equality of performances).

# ## Variants of Inputec

# ### M2 - Cost effective Inputec
# 
# #### 20 configurations per videos instead of 201 

# In[66]:

print("First variant of Inputec - with less configurations")

nb_config_ce = 20

def gen_dataset_ce(inputs_names):

    res = pd.DataFrame(np.zeros(nb_config_ce*len(inputs_names)*nb_col)
                       .reshape(nb_config_ce*len(inputs_names), nb_col))
    res.columns = name_col

    for i in range(len(inputs_names)):
        video_name = inputs_names[i]
        index_video = np.where(np.array([v[:-4] for v in v_names], str)==video_name)[0][0]
        sizes = listVideo[index_video][predDimension]
        video_prop = np.array(meta_perf.loc[video_name][1:], float)
        moy = np.mean(sizes)
        std = np.std(sizes)
        for config_id in range(nb_config_ce):
            #config_rank = sorted_config[config_id]
            val = list(tuple(video_prop) + tuple(val_config.loc[config_id]))
            val.append((sizes[config_id]-moy)/std)
            res.loc[i*nb_config_ce+config_id] = val
    return res

training_data_ce = gen_dataset_ce(train_index)
test_data_ce = gen_dataset_ce(test_index)

print("Training size : ", training_data_ce.shape[0])
print("Test size : ", test_data_ce.shape[0])

test_data_ce


# In[67]:


X_train_ce = np.array(training_data_ce.drop(["bitrate"],axis=1), float)
y_train_ce = np.array(training_data_ce["bitrate"], float)

X_test_ce = np.array(test_data_ce.drop(["bitrate"],axis=1), float)
y_test_ce = np.array(test_data_ce["bitrate"], float)


# In[68]:


LA_ce = RandomForestRegressor()

LA_ce.fit(X_train_ce, y_train_ce)

y_pred_ce = LA_ce.predict(X_test_ce)


# In[69]:


print(LA_ce.feature_importances_)


# In[70]:


np.mean(np.abs(y_pred_ce-y_test_ce))


# In[71]:


print(name_col)


# In[72]:


# the performance values for the configuration chosen by Inputec, for the test set of videos
inputec_m2 = []

# the performance rankings for the configuration chosen by Inputec, for the test set of videos
inputec_m2_ranks = []

# for each video in the test set
for ti in test_index:
    # we retrieve the test index
    index_video = np.where(np.array([v[:-4] for v in v_names], str)==ti)[0][0]
    # list of predictions for inputec
    pred_inputec = []
    # for each configuration
    for i in range(nb_config):
        # we add input properties to the configurations
        video_prop = list(tuple(meta_perf.loc[ti][1:])+tuple(val_config.loc[i]))
        # we predict the value associated to the configuration
        pred_inputec.append(LA_ce.predict(np.array(video_prop, float).reshape(1, 33)))
    # sorted performances
    sorted_perfs = sorted(listVideo[index_video][predDimension])
    # the performance of the configuration chosen by Inputec
    perf = listVideo[index_video].loc[np.argmin(pred_inputec)][predDimension]
    # we add it to the perf array
    inputec_m2.append(perf)
    # the related ranking (between 0 and 200, hopefully close to 0)
    inputec_m2_ranks.append(find_rank(sorted_perfs, perf))


# In[73]:


pd.Series(inputec_m2).describe()


# In[74]:

print("First variant of Inputec - ranks distribution results")
print(pd.Series(inputec_m2_ranks).describe())

print("First variant of Inputec - Done !")

print("Second variant of Inputec - drop unaffordable input properties")

# ### M3 - Property selection
# 
# #### Only keep affordable properties in the model - drop SLEEQ MOS and Banding MOS

# In[75]:


name_col = list(meta_perf.columns)[3:]

for vcc in val_config.columns:
    name_col.append(vcc)

name_col.append("bitrate")

nb_col = len(name_col)
nb_config = 201

def gen_dataset(inputs_names):
    
    res = pd.DataFrame(np.zeros(nb_config*len(inputs_names)*nb_col).reshape(nb_config*len(inputs_names), nb_col))
    res.columns = name_col

    for i in range(len(inputs_names)):
        video_name = inputs_names[i]
        index_video = np.where(np.array([v[:-4] for v in v_names], str)==video_name)[0][0]
        sizes = listVideo[index_video][predDimension]
        sorted_config = sorted(range(len(sizes)), key=lambda k: sizes[k])
        video_prop = np.array(meta_perf.loc[video_name][3:], float)
        moy = np.mean(sizes)
        std = np.std(sizes)            
        for config_id in range(len(sorted_config)):
            config_rank = sorted_config[config_id]
            val = list(tuple(video_prop) + tuple(val_config.loc[config_id]))
            val.append((sizes[config_id]-moy)/std)
            res.loc[i*nb_config+config_id] = val
    return res

training_data_m3 = gen_dataset(train_index)
test_data_m3 = gen_dataset(test_index)

print("Training size : ", training_data_m3.shape[0])
print("Test size : ", test_data_m3.shape[0])

test_data_m3


# In[76]:


X_train_m3 = np.array(training_data_m3.drop(["bitrate"],axis=1), float)
y_train_m3 = np.array(training_data_m3["bitrate"], float)

X_test_m3 = np.array(test_data_m3.drop(["bitrate"],axis=1), float)
y_test_m3 = np.array(test_data_m3["bitrate"], float)


# In[77]:


LA_m3 = RandomForestRegressor()

LA_m3.fit(X_train_m3, y_train_m3)


# In[78]:


y_pred_m3 = LA_m3.predict(X_test_m3)


# In[79]:


print(LA_m3.feature_importances_)


# In[80]:


np.mean(np.abs(y_pred_m3-y_test_m3))


# In[81]:


# the performance values for the configuration chosen by Inputec, for the test set of videos
inputec_m3 = []

# the performance rankings for the configuration chosen by Inputec, for the test set of videos
inputec_m3_ranks = []

# for each video in the test set
for ti in test_index:
    # we retrieve the test index
    index_video = np.where(np.array([v[:-4] for v in v_names], str)==ti)[0][0]
    # list of predictions for inputec
    pred_inputec = []
    # for each configuration
    for i in range(nb_config):
        # we add input properties to the configurations
        video_prop = list(tuple(meta_perf.loc[ti][3:])+tuple(val_config.loc[i]))
        # we predict the value associated to the configuration
        pred_inputec.append(LA_m3.predict(np.array(video_prop, float).reshape(1, 31)))
    # sorted performances
    sorted_perfs = sorted(listVideo[index_video][predDimension])
    # the performance of the configuration chosen by Inputec
    perf = listVideo[index_video].loc[np.argmin(pred_inputec)][predDimension]
    # we add it to the perf array
    inputec_m3.append(perf)
    # the related ranking (between 0 and 200, hopefully close to 0)
    inputec_m3_ranks.append(find_rank(sorted_perfs, perf))


# In[82]:


pd.Series(inputec_m3).describe()


# In[83]:
print("Second variant of Inputec - ranks distribution results")
print(pd.Series(inputec_m3_ranks).describe())

print("Second variant of Inputec - Done !")

# In[ ]:





# In[ ]:


print("DONE!!! Thanks for waiting, thanks for testing our artifact :-)")
print("You can check the results directory (see code/results)")

# In[ ]:





