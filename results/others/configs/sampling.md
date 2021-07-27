# Testing the uniformity of sampled configurations

To select the configuration options, we read the documentation of each system. 

We manually extract the options affecting the performances of the system according to the documentation. 

We then sample \#C configurations by using a random sampling. 

We check the uniformity of the different option values with a Kolmogorov-Smirnov test applied to each configuration option. 

This notebook details the different results related to these tests, for each system and each set of configurations kept for this system.


## About the Kolmogorov-Smirnov test

We used one-sample Kolmogorov-Smirnov tests to compare the cumulative distributive function (aka cdf) of the generated distribution of values for configuration options to the theorical uniform distribution.

The idea of this test is to study the maximal difference of two cdfs, (i.e. $max_{x}|F_{1}(x) - F_{2}(x)|$, where $F_{1}$ and $F_{2}$ are the two cdfs). The bigger the difference, the more the distributions are far from each other.

The one-sample version of this test compares the empirical cdf ($F_{1} =  F_{emp}$) to the theorical cdf ($F_{2} =  F_{t}$). Here, we want the theorical law to be uniform. Let us say that an option has ten ordered values $ k = 1 ... 10$. Then, the empirical cdf respects $\forall k \in [|1,10|], F_{emp}(k) =  c(k)*\frac{1}{\#(config)}$, where c(k) is the count of ``k`` value occurences, while the theorical cdf respects $\forall k \in [|1,10|], F_{t}(k) = P(X<=k) = \frac{k}{10}$.

We choose the threshold $0.05$. The null hypothesis of same distribution (i.e. the sampling can be considered as uniform) is rejected if $pval < 0.05$.

## Code

#### First, we import some libraries


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
# Decision Tree
from sklearn.tree import DecisionTreeRegressor, plot_tree
# To add interactions in linear regressions models
from sklearn.preprocessing import PolynomialFeatures
# Elasticnet is an hybrid method between ridge and Lasso
from sklearn.linear_model import LinearRegression, ElasticNet
# To separate the data into training and test
from sklearn.model_selection import train_test_split
# Simple clustering (iterative steps)
from sklearn.cluster import KMeans
# get interactions of features
from sklearn.preprocessing import PolynomialFeatures


# we use it to interact with the file system
import os
# compute time
from time import time

# statistics
import scipy.stats as sc
# hierarchical clustering, clusters
from scipy.cluster.hierarchy import linkage, cut_tree, leaves_list
from scipy import stats
# statistical tests
from scipy.stats import mannwhitneyu

# no warning
import warnings
warnings.filterwarnings("ignore")
```

#### Then the data


```python
data_dir = "../../../data/"
name_systems = ["nodejs", "poppler", "xz", "x264", "gcc", "lingeling", "sqlite", "imagemagick"]

data = dict()
inputs_name = dict()
inputs_count = dict()

inputs_perf = dict()

inputs_perf["gcc"] = ["size", "ctime", "exec"]
inputs_perf["lingeling"] = ["conflicts", "cps", "reductions"]
inputs_perf["nodejs"] = ["ops"]
inputs_perf["poppler"] = ["size", "time"]
inputs_perf["sqlite"] = ["q"+str(i+1) for i in range(15)]
inputs_perf["x264"] = ["size", "kbs", "fps", "etime", "cpu"]
inputs_perf["xz"] = ["size", "time"]


inputs_feat = dict()

inputs_feat["gcc"] = ["optim","-floop-interchange","-fprefetch-loop-arrays","-ffloat-store","-fno-asm"]
inputs_feat["imagemagick"] = ["-limit memory","-posterize","-gaussian-blur","-limit thread","-quality"]
inputs_feat["lingeling"] = ["--boost", "--carduse", "--decompose", "--gluescale", "--lkhd", "--memlim", 
"--minimize", "--prbsimple", "--sweepirr", "--sweepred"]
inputs_feat["nodejs"] = ["--jitless", "--experimental-wasm-modules", "--experimental-vm-modules",
                         "--preserve-symlinks-main","--no-warnings","--node-memory-debug"]
inputs_feat["poppler"] = ["format","j","jp2","jbig2","ccitt"]
inputs_feat["sqlite"] = ["-deserialize", "-memtrace", "-maxsize", "-append", "-output"]
inputs_feat["x264"] = ["cabac", "ref", "deblock", "analyse", "me", "subme", "mixed_ref", "me_range", "trellis", 
                "8x8dct", "fast_pskip", "chroma_qp_offset", "bframes", "b_pyramid", "b_adapt", "direct", 
                "weightb", "open_gop", "weightp", "scenecut", "rc_lookahead", "mbtree", "qpmax", "aq-mode"]
inputs_feat["xz"] = ["memory","format","level","depth"]


inputs_categ = dict()

inputs_categ["gcc"] = ["optim"]
inputs_categ["lingeling"] = []
inputs_categ["nodejs"] = []
inputs_categ["poppler"] = ["format"]
inputs_categ["sqlite"] = []
inputs_categ["x264"] = ['analyse', 'me', 'direct', 'deblock']
inputs_categ["xz"] = ['memory', 'format']


for ns in name_systems:
    
    data_path = data_dir+ns+'/'
    
    inputs = sorted(os.listdir(data_path))
    inputs.remove('others')

    inputs_name[ns] = inputs
    inputs_count[ns] = len(inputs)
    
    for i in range(len(inputs)):
        loc = data_path+inputs[i]
        data[ns, i] = pd.read_csv(loc)
```

###### First a function  to replace the categorical values with numericals one


```python
def replace_with_numerical_values(arr):
    
    ### Input :
    #### arr an array of string, whatever the strings
    
    ### Output :
    #### num_arr, with numerical values corresponding to the string values
    
    #### eg if arr is ['a', 'c', 'b', 'a', 'b'], arr_num is [1, 2, 3, 1, 3]
    
    #### we first isolate the values present in the array arr
    #### in our little example, values would be equal to ['a', 'b', 'c'] 
    values = pd.Series(ma).unique()
    
    #### a list of list, for each value, the list of indexes for this value
    index_values = [np.where(ma==val)[0] for val in values]

    #### init the resulting array
    num_arr = np.zeros(len(ma))

    #### replace with the actual indexes
    for i in range(len(index_values)):
        for ind_val in index_values[i]:
            num_arr[ind_val] = i+1
    
    #### returning the array with numerical values
    return num_arr

### Testing with an example 
ma = data["x264", 1][inputs_categ["x264"][1]]
print(ma)
replace_with_numerical_values(ma)
```

    0       dia
    1       hex
    2       hex
    3       umh
    4       hex
           ... 
    196     hex
    197     hex
    198     hex
    199     hex
    200    tesa
    Name: me, Length: 201, dtype: object





    array([1., 2., 2., 3., 2., 1., 4., 3., 2., 2., 2., 1., 2., 2., 3., 2., 2.,
           2., 2., 2., 2., 2., 1., 2., 2., 2., 2., 2., 2., 1., 1., 1., 1., 1.,
           1., 1., 1., 2., 2., 2., 2., 4., 4., 2., 1., 2., 2., 2., 3., 2., 2.,
           2., 2., 2., 2., 1., 3., 2., 2., 2., 2., 2., 3., 3., 3., 3., 1., 3.,
           3., 3., 3., 2., 2., 1., 3., 3., 3., 2., 3., 3., 3., 3., 3., 3., 3.,
           3., 3., 2., 1., 3., 3., 3., 3., 3., 3., 3., 3., 2., 2., 1., 4., 4.,
           4., 4., 4., 4., 4., 2., 4., 4., 1., 1., 4., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 1.,
           1., 1., 1., 2., 2., 3., 4., 1., 1., 2., 1., 1., 1., 4., 1., 1., 1.,
           2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 4., 4., 2., 1., 1.,
           1., 1., 2., 2., 2., 2., 2., 2., 3., 2., 2., 2., 2., 2., 2., 2., 2.,
           3., 2., 3., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 4.])



#### Then, a function to check the uniformity for a software system


```python
def check_uniformity(ns):
    
    ### Input:
    #### ns, the name of the software system
    
    ### Output:
    #### Print the list of options, and says if it is uniform or not based on the configuration space
    
    # first, we define the dataframe we are working with
    # the first dataframe with configurations, they are all the same anyway in terms of configurations
    df = data[ns, 0]
    
    # we store the number of lines in the dataframe, nbconfig
    nb_config = df.shape[0]
    
    # First, we replace the values of options that are neither boolean nor numericals
    option_names = inputs_feat[ns]
    
    for opt_name in option_names:
        # we use the previous function on the distribution of performance
        # perf = replace_with_numerical_values(df[opt_name])
        perf = np.copy(df[opt_name])
        # we count the number per value
        opt_values_count = pd.Series(perf).value_counts()
        # we normalize
        opt_values_count /= nb_config
        # we initiate the cumulative distributive function array, to store its values
        cdf_values = []
        # the sum of previous values
        s = 0
        # we add each value separately, to the total sum and to the array
        for opt_val_count in opt_values_count:
            s+=opt_val_count
            cdf_values.append(s)
        # finally we add 1 to the values, it is the end of the array
        cdf_values.append(1)
        # we apply the Kolmogorov-Smirnov test
        pval = stats.kstest(cdf_values, 'uniform').pvalue
        # we display the results
        print(opt_name, ", p-val = ", str(round(pval,3))+",", (pval>=0.05)*"Uniform sampling")
```

## Results for each software system

### gcc


```python
check_uniformity("gcc")
```

    optim , p-val =  0.423, Uniform sampling
    -floop-interchange , p-val =  0.074, Uniform sampling
    -fprefetch-loop-arrays , p-val =  0.074, Uniform sampling
    -ffloat-store , p-val =  0.074, Uniform sampling
    -fno-asm , p-val =  0.074, Uniform sampling


Ok for gcc!

### imagemagick


```python
check_uniformity("imagemagick")
```

    -limit memory , p-val =  0.423, Uniform sampling
    -posterize , p-val =  0.309, Uniform sampling
    -gaussian-blur , p-val =  0.188, Uniform sampling
    -limit thread , p-val =  0.259, Uniform sampling
    -quality , p-val =  0.188, Uniform sampling


Ok for imagemagick

### lingeling


```python
check_uniformity("lingeling")
```

    --boost , p-val =  0.074, Uniform sampling
    --carduse , p-val =  0.188, Uniform sampling
    --decompose , p-val =  0.074, Uniform sampling
    --gluescale , p-val =  0.423, Uniform sampling
    --lkhd , p-val =  0.2, Uniform sampling
    --memlim , p-val =  0.074, Uniform sampling
    --minimize , p-val =  0.074, Uniform sampling
    --prbsimple , p-val =  0.188, Uniform sampling
    --sweepirr , p-val =  0.188, Uniform sampling
    --sweepred , p-val =  0.157, Uniform sampling


Ok for lingeling

### nodeJS


```python
check_uniformity("nodejs")
```

    --jitless , p-val =  0.074, Uniform sampling
    --experimental-wasm-modules , p-val =  0.074, Uniform sampling
    --experimental-vm-modules , p-val =  0.074, Uniform sampling
    --preserve-symlinks-main , p-val =  0.074, Uniform sampling
    --no-warnings , p-val =  0.074, Uniform sampling
    --node-memory-debug , p-val =  0.074, Uniform sampling


Ok for nodejs

### poppler


```python
check_uniformity("poppler")
```

    format , p-val =  0.074, Uniform sampling
    j , p-val =  0.074, Uniform sampling
    jp2 , p-val =  0.074, Uniform sampling
    jbig2 , p-val =  0.074, Uniform sampling
    ccitt , p-val =  0.074, Uniform sampling


Ok for poppler!

### SQLite


```python
check_uniformity("sqlite")
```

    -deserialize , p-val =  0.074, Uniform sampling
    -memtrace , p-val =  0.074, Uniform sampling
    -maxsize , p-val =  0.0, 
    -append , p-val =  0.074, Uniform sampling
    -output , p-val =  0.0, 



```python
data["sqlite", 1]["-maxsize"].unique()
```




    array([0])




```python
data["sqlite", 1]["-output"].unique()
```




    array([0])



for SQLite, there are problems with memtrace and output -> they always have tha same value at 0.
It explains why they are not changing, to take into account when reading RQ2 and RQ3

### x264


```python
check_uniformity("x264")
```

    cabac , p-val =  0.065, Uniform sampling
    ref , p-val =  0.084, Uniform sampling
    deblock , p-val =  0.051, Uniform sampling
    analyse , p-val =  0.061, Uniform sampling
    me , p-val =  0.081, Uniform sampling
    subme , p-val =  0.556, Uniform sampling
    mixed_ref , p-val =  0.074, Uniform sampling
    me_range , p-val =  0.026, 
    trellis , p-val =  0.155, Uniform sampling
    8x8dct , p-val =  0.018, 
    fast_pskip , p-val =  0.018, 
    chroma_qp_offset , p-val =  0.074, Uniform sampling
    bframes , p-val =  0.044, 
    b_pyramid , p-val =  0.005, 
    b_adapt , p-val =  0.083, Uniform sampling
    direct , p-val =  0.099, Uniform sampling
    weightb , p-val =  0.015, 
    open_gop , p-val =  0.014, 
    weightp , p-val =  0.11, Uniform sampling
    scenecut , p-val =  0.008, 
    rc_lookahead , p-val =  0.179, Uniform sampling
    mbtree , p-val =  0.074, Uniform sampling
    qpmax , p-val =  0.0, 
    aq-mode , p-val =  0.062, Uniform sampling


Again, for x264 there are some options me_range, 8x8dct, bframes, b_pyramid, weightb, open_gop, and qpmax, that did not pass the test.


```python
df = data["x264",1]

val = ["me_range", "8x8dct", "bframes", "b_pyramid", "weightb", "open_gop", "qpmax"]

for v in val:
    print(df[v].value_counts())

```

    16    154
    24     47
    Name: me_range, dtype: int64
    1    159
    0     42
    Name: 8x8dct, dtype: int64
    3     115
    0      37
    16     25
    8      24
    Name: bframes, dtype: int64
    2       155
    None     38
    1         8
    Name: b_pyramid, dtype: int64
    1       162
    None     39
    Name: weightb, dtype: int64
    0       163
    None     38
    Name: open_gop, dtype: int64
    69    201
    Name: qpmax, dtype: int64


- For me_range, too many "24" values

- For 8x8dct, too many "1" values

- For bframes, too many "3" values

- For b_pyramid, too many "2" values

- For weightb, too many "0" values

- For qpmax, one value for all the array

It is OK since these values are not influential, but we will be careful when commenting the results

### xz


```python
check_uniformity("xz")
```

    memory , p-val =  0.188, Uniform sampling
    format , p-val =  0.074, Uniform sampling
    level , p-val =  0.162, Uniform sampling
    depth , p-val =  0.188, Uniform sampling


for xz, OK


```python

```
