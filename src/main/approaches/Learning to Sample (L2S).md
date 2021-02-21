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

#### Libraries


```python
# for arrays
import numpy as np

# for dataframes
import pandas as pd

# plots
import matplotlib.pyplot as plt
%matplotlib inline
# high-level plots
import seaborn as sns

import statsmodels.api as sm

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


from learner.mlearner import learn_with_interactions, learn_without_interactions, sample_random, stepwise_feature_selection
from learner.model import genModelTermsfromString, Model, genModelfromCoeff

import warnings
warnings.filterwarnings("ignore")
```

#### implementation of the approach


```python
class L2S:
    
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
    
    ### Step 1: Extraction Process of Performance Models
    
    #Select a good model for predicting the performance of the source video
    
    #Original files:
    #https://github.com/cmu-mars/model-learner/blob/tutorial/learner/mlearner.py for the stepwise selection
    #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html for the interactions
    
    # @PooyanJamshidi:
    # We just change slightly some functions from the original repository,
    # mainly because we don't want to add a constant in the model
    # + steps 2 and 3 were implemented in matlab but we did not find them in python
    def stepwise_selection(self, X, y,
                           initial_list = [], 
                           threshold_in = 0.01, 
                           threshold_out = 0.05, 
                           verbose=False):

        ndim = X.shape[1]
        features = [i for i in range(ndim)]
        included = list(initial_list)

        while True:
            changed=False

            # forward step (removed a constant)
            excluded = list(set(features)-set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(y, pd.DataFrame(X[included+[new_column]])).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if verbose:
                    print('Add {:30} with p-value {:.5}'.format(best_feature, best_pval))

            # backward step
            model = sm.OLS(y, pd.DataFrame(X[included])).fit()
            pvalues = model.pvalues
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.5}'.format(worst_feature, worst_pval))
            if not changed:
                if verbose:
                    print("Construction of the model completed!")
                break
        return included
    
    ### Step 2: Active Sampling
    
    #### A - ] Exploitation : use the source's prediction model
    
    ##### (i) Sort the coefficients of the previous constructed model
    ##### (ii) Choose the coefficient with the highest value
    ##### (iii) Select the configurations with this feature activated

    # I assumed it was recursive, with a decreasing influence in the selection 
    # for a decreasing importance in the regression.
    
    def select_exploitation(self, df, sc, config_selected):
        
        self.nb_config = int(self.nb_config_exploitation - len(config_selected))
        
        if self.nb_config == 0:
            #print("Done!\n")
            return config_selected

        # if we don't have any important coefficient left to help us choose configs
        # we take the nb_config first configurations
        if len(sc) == 0:
            #print("Selecting " + str(self.nb_config) + " configurations from the rest of the dataset!")
            for conf in df.index[0:self.nb_config]:
                config_selected.append(conf)
            return config_selected

        # otherwise we just use the best coef to choose configs
        else:

            # we choose the best features coef (biggest absolute value)
            most_important_coef = sc[0]

            #print("Feature : " + str(most_important_coef))

            # configs with this feature activated
            imp_index = np.where(df[most_important_coef]==1)[0]

            # number of configs with this feature activated
            nb_imp_index = len(imp_index)

            # if we have more values to choose 
            # than the number of configurations with the best feature activated
            # we add all the configuration to the selected set
            # and we select the rest of the configuration based on other coefficients
            if nb_imp_index <= self.nb_config:
                for conf in df.iloc[imp_index].index:
                    config_selected.append(conf)
                #if nb_imp_index > 0:
                #    print("Added "+str(nb_imp_index)+ " values, "+
                #          str(self.nb_config-nb_imp_index)+" left to choose \n")
                # then we apply recursively this method to the rest of the dataframe
                return self.select_exploitation(df.iloc[np.where(df[most_important_coef]==0)[0]], 
                                              sc[1:len(sc)],
                                              config_selected)

            # otherwise we have enough values with this features activated
            # to select all the remaining configurations
            # so we apply the method to the dataframe containing all the feature activated
            # and we select the configuration by using the followings features
            else:
                return self.select_exploitation(df.iloc[imp_index], 
                                     sc[1:len(sc)], 
                                     config_selected)
    
    
    
    #### B-] Exploration : Select specific configurations, similar between the source and the target
    
    # I choose to select the group in one step:
    # if you select config per config, you may choose a local optimal
    
    def select_exploration(self, exploitation_conf, ratio_exploitation, number_group = 10):
        
        nb_exploration = int(np.round(self.config_tot*(1-ratio_exploitation)))

        #target = self.listVideo[id_target]

        # all the config left for exploration
        # total minus those chosen for exploitation
        explor_conf = np.setdiff1d(self.source.index, exploitation_conf)

        # initialization : we take the first nb_exploration config
        best_explor = explor_conf[0:nb_exploration]

        # we group it with the exploitation configurations
        conf = np.concatenate((exploitation_conf, best_explor), axis=0)
        
        # for the moment, it's our best entropy
        best_entropy = sc.entropy(self.target.iloc[conf][self.predDimension], 
                                  self.source.iloc[conf][self.predDimension])

        # then we incrementally select the configurations to diminish the entropy 
        group_counter = 0

        while group_counter < number_group:

            group_counter +=1

            # current group to 'challenge' the best result
            np.random.shuffle(explor_conf)
            current_explor = explor_conf[0:nb_exploration]

            # we group it with the exploitation configurations
            conf = np.concatenate((exploitation_conf, current_explor), axis=0)

            # we compute the Kullback Leibler divergence between the source and the target
            current_entropy = sc.entropy(self.target.iloc[conf][self.predDimension], 
                                         self.source.iloc[conf][self.predDimension])

            # we finally take the group giving the lowest entropy
            # if this group is better than the best group, we replace it by the new one
            if current_entropy > best_entropy:
                #print("Entropy gained : "+str(current_entropy-best_entropy))
                best_entropy = current_entropy
                best_explor = current_explor

        return best_explor
    
    
    
    def learn(self, source_id, target_id, ratio_exploitation = 0.3, 
              l2s_tr_ratio = 0.8, 
              train_size = 20,
              learning_algorithm = RandomForestRegressor):

        # the source video
        self.source = self.listVideo[source_id]

        # the number of config used in the training
        self.config_tot = int(train_size)
        
        if train_size <= 1:
            self.config_tot = int(train_size*self.source.shape[1])

        # transform some variables into dummies, to fit the orginal paper
        # since we don't want to introduce a meaningless constant in the model, 
        # we have to keep all columns

        X_src = pd.DataFrame(np.array(self.source.drop([self.predDimension], axis = 1), dtype=int))

        #X_src = self.source[self.keep_features]
        
        # add interactions
        poly = PolynomialFeatures(degree = 1, interaction_only = True, include_bias = True)
        
        # degree 2 take too much time + it will not scale for large configuration spaces...
        # IMO O(n) or O(nlog(n)) are the complexity we should all target for our algorithms
        
        X_interact = pd.DataFrame(np.array(poly.fit_transform(X_src), int))

        # performance variable, to predict
        y_src = self.source[self.predDimension]
        
        # we train the model with the training data
        
        # print("\n############### I- Knowledge extraction #################\n")

        selected_features = self.stepwise_selection(X_interact, y_src)

        # print("\n############### II- Sampling #################\n")

        reg = LinearRegression()

        reg.fit(X_interact[selected_features], y_src)

        sorted_coefs = pd.Series(np.abs(reg.coef_), 
                                 selected_features, 
                                 dtype='float64').sort_values(ascending=False).index

        # print("A- EXPLOITATION\n")
        
        self.nb_config_exploitation = int(ratio_exploitation*self.config_tot)
        
        exploitation_conf = self.select_exploitation(X_interact, sorted_coefs, [])
        
        # print(exploitation_conf)

        # print("\nB- EXPLORATION\n")

        # we ensure we sample the configurations of the training set
        # which removes the potential threat of using the configuration of the testing set
        # during the training
        
        # target
        self.target = self.listVideo[target_id]
        
        exploration_conf = self.select_exploration(exploitation_conf, ratio_exploitation)

        sampled_conf = np.concatenate((exploitation_conf,exploration_conf), axis=0)
        
        # print(sampled_conf)
        
        # print("\n############### III- Performance Model Learning #################\n")

        # we build a performance model for the target
        # instead of using all the configurations, we use the sampled configuration
        # ie we remove the unnecessary configurations
        # print(len(sampled_conf))
        
        X_tgt = self.target.drop([self.predDimension], axis = 1)
        y_tgt = self.target[self.predDimension]

        X_tgt_tr = X_tgt.iloc[sampled_conf]
        y_tgt_tr = y_tgt[sampled_conf]
        
        #X_tgt_te = self.target[self.keep_features].drop(sampled_conf, inplace = False, axis=0)
        #y_tgt_te = self.target[self.predDimension].drop(sampled_conf, inplace = False, axis=0)

        # The shift function, to transfer the prediction from the source to the target
        lf = learning_algorithm()
        lf.fit(X_tgt_tr, y_tgt_tr)
        y_tgt_pred = lf.predict(X_tgt)
        
        # We return the mean average percentage error 
        # between the real values of y_test from target 
        # and the predictions shifted 
        return self.mse(y_tgt, y_tgt_pred)
    
    def predict_conf(self, source_id, target_id, ratio_exploitation = 0.3, 
              l2s_tr_ratio = 0.8, 
              train_size = 20,
              learning_algorithm = XGBRegressor):

        # the source video
        self.source = self.listVideo[source_id]

        # the number of config used in the training
        self.config_tot = int(train_size)
        
        if train_size <= 1:
            self.config_tot = int(train_size*self.source.shape[1])

        # transform some variables into dummies, to fit the orginal paper
        # since we don't want to introduce a meaningless constant in the model, 
        # we have to keep all columns

        X_src = pd.DataFrame(np.array(self.source.drop([self.predDimension], axis = 1), dtype=int))

        #X_src = self.source[self.keep_features]
        
        # add interactions
        poly = PolynomialFeatures(degree = 1, interaction_only = True, include_bias = True)
        
        # degree 2 take too much time + it will not scale for large configuration spaces...
        # IMO O(n) or O(nlog(n)) are the complexity we should all target for our algorithms
        
        X_interact = pd.DataFrame(np.array(poly.fit_transform(X_src), int))

        # performance variable, to predict
        y_src = self.source[self.predDimension]
        
        # we train the model with the training data
        
        # print("\n############### I- Knowledge extraction #################\n")

        selected_features = self.stepwise_selection(X_interact, y_src)

        # print("\n############### II- Sampling #################\n")

        reg = LinearRegression()

        reg.fit(X_interact[selected_features], y_src)

        sorted_coefs = pd.Series(np.abs(reg.coef_), 
                                 selected_features, 
                                 dtype='float64').sort_values(ascending=False).index

        # print("A- EXPLOITATION\n")
        
        self.nb_config_exploitation = int(ratio_exploitation*self.config_tot)
        
        exploitation_conf = self.select_exploitation(X_interact, sorted_coefs, [])
        
        # print(exploitation_conf)

        # print("\nB- EXPLORATION\n")

        # we ensure we sample the configurations of the training set
        # which removes the potential threat of using the configuration of the testing set
        # during the training
        
        # target
        self.target = self.listVideo[target_id]
        
        exploration_conf = self.select_exploration(exploitation_conf, ratio_exploitation)

        sampled_conf = np.concatenate((exploitation_conf,exploration_conf), axis=0)
        
        #print("\n############### III- Performance Model Learning #################\n")

        # we build a performance model for the target
        # instead of using all the configurations, we use the sampled configuration
        # ie we remove the unnecessary configurations
        # print(len(sampled_conf))
        
        X_tgt = self.target.drop([self.predDimension], axis = 1)
        y_tgt = self.target[self.predDimension]

        X_tgt_tr = X_tgt.iloc[sampled_conf]
        y_tgt_tr = y_tgt[sampled_conf]
        
        #X_tgt_te = self.target[self.keep_features].drop(sampled_conf, inplace = False, axis=0)
        #y_tgt_te = self.target[self.predDimension].drop(sampled_conf, inplace = False, axis=0)

        # The shift function, to transfer the prediction from the source to the target
        lf = learning_algorithm()
        lf.fit(X_tgt_tr, y_tgt_tr)
        y_tgt_pred = lf.predict(X_tgt)
        
        # We return the mean average percentage error 
        # between the real values of y_test from target 
        # and the predictions shifted 
        return np.argmin(y_tgt_pred)
```

#### Learning algorithm


```python
l2s = L2S()

LAs = [DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, SVR]
for i in range(5):
    source_id = np.random.randint(0,1000)
    target_id = np.random.randint(0,1000)
    for la in LAs:
        print(la, l2s.learn(source_id = source_id, 
                            target_id = target_id, 
                            train_size = 20, 
                            learning_algorithm=la))
```

    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 3281685.2859895527
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 2042005.232342937
    <class 'xgboost.sklearn.XGBRegressor'> 3062540.786840061
    <class 'sklearn.svm._classes.SVR'> 9576159.427733865
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 27405524.146231342
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 6519916.737222477
    <class 'xgboost.sklearn.XGBRegressor'> 7263449.448959386
    <class 'sklearn.svm._classes.SVR'> 71663105.14875856
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 10962.400734825867
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 6764.398971992097
    <class 'xgboost.sklearn.XGBRegressor'> 2998.9533698724767
    <class 'sklearn.svm._classes.SVR'> 61580.4388545609
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 1613.7505716417907
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 2706.459407840895
    <class 'xgboost.sklearn.XGBRegressor'> 5054.611589117543
    <class 'sklearn.svm._classes.SVR'> 16379.99248684554
    <class 'sklearn.tree._classes.DecisionTreeRegressor'> 1783955.5832855713
    <class 'sklearn.ensemble._forest.RandomForestRegressor'> 1267925.4245734932
    <class 'xgboost.sklearn.XGBRegressor'> 618574.6346787198
    <class 'sklearn.svm._classes.SVR'> 6327715.289539618


#### Chosen algorithm :  XGBRegressor (however it may depends on the choice of videos)

Bug with Linear Regression, mse too low? To analyze

Again, it depends on the video we consider

#### Predictions

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
l2s = L2S()
l2s_confs = dict()
for i in range(len(index_test)):
    it = index_test[i]
    source_index_train = np.random.randint(0, len(v_names_train))
    source_id = index_train[source_index_train]
    for ts in train_sizes:
        l2s_confs[(i, ts)] = l2s.predict_conf(source_id = source_id, target_id = it, train_size=ts,
                                      learning_algorithm = XGBRegressor)
```


```python
l2s_confs
```




    {(0, 5): 1,
     (0, 10): 34,
     (0, 15): 153,
     (0, 20): 123,
     (0, 25): 166,
     (0, 30): 123,
     (1, 5): 57,
     (1, 10): 169,
     (1, 15): 93,
     (1, 20): 189,
     (1, 25): 100,
     (1, 30): 89,
     (2, 5): 111,
     (2, 10): 163,
     (2, 15): 176,
     (2, 20): 190,
     (2, 25): 190,
     (2, 30): 161,
     (3, 5): 85,
     (3, 10): 4,
     (3, 15): 32,
     (3, 20): 161,
     (3, 25): 194,
     (3, 30): 155,
     (4, 5): 76,
     (4, 10): 91,
     (4, 15): 10,
     (4, 20): 175,
     (4, 25): 42,
     (4, 30): 79,
     (5, 5): 8,
     (5, 10): 87,
     (5, 15): 103,
     (5, 20): 100,
     (5, 25): 97,
     (5, 30): 189,
     (6, 5): 49,
     (6, 10): 168,
     (6, 15): 166,
     (6, 20): 100,
     (6, 25): 159,
     (6, 30): 168,
     (7, 5): 48,
     (7, 10): 189,
     (7, 15): 3,
     (7, 20): 98,
     (7, 25): 92,
     (7, 30): 92,
     (8, 5): 19,
     (8, 10): 85,
     (8, 15): 181,
     (8, 20): 170,
     (8, 25): 179,
     (8, 30): 181,
     (9, 5): 0,
     (9, 10): 62,
     (9, 15): 178,
     (9, 20): 96,
     (9, 25): 109,
     (9, 30): 189,
     (10, 5): 8,
     (10, 10): 62,
     (10, 15): 123,
     (10, 20): 91,
     (10, 25): 91,
     (10, 30): 96,
     (11, 5): 4,
     (11, 10): 178,
     (11, 15): 123,
     (11, 20): 169,
     (11, 25): 165,
     (11, 30): 197,
     (12, 5): 92,
     (12, 10): 100,
     (12, 15): 85,
     (12, 20): 96,
     (12, 25): 123,
     (12, 30): 189,
     (13, 5): 29,
     (13, 10): 74,
     (13, 15): 62,
     (13, 20): 171,
     (13, 25): 200,
     (13, 30): 171,
     (14, 5): 6,
     (14, 10): 168,
     (14, 15): 137,
     (14, 20): 153,
     (14, 25): 184,
     (14, 30): 154,
     (15, 5): 1,
     (15, 10): 103,
     (15, 15): 87,
     (15, 20): 75,
     (15, 25): 192,
     (15, 30): 65,
     (16, 5): 1,
     (16, 10): 78,
     (16, 15): 169,
     (16, 20): 109,
     (16, 25): 93,
     (16, 30): 101,
     (17, 5): 166,
     (17, 10): 168,
     (17, 15): 166,
     (17, 20): 166,
     (17, 25): 157,
     (17, 30): 168,
     (18, 5): 20,
     (18, 10): 83,
     (18, 15): 25,
     (18, 20): 123,
     (18, 25): 192,
     (18, 30): 169,
     (19, 5): 4,
     (19, 10): 63,
     (19, 15): 189,
     (19, 20): 85,
     (19, 25): 97,
     (19, 30): 90,
     (20, 5): 8,
     (20, 10): 83,
     (20, 15): 169,
     (20, 20): 89,
     (20, 25): 91,
     (20, 30): 168,
     (21, 5): 123,
     (21, 10): 156,
     (21, 15): 100,
     (21, 20): 94,
     (21, 25): 85,
     (21, 30): 170,
     (22, 5): 97,
     (22, 10): 181,
     (22, 15): 20,
     (22, 20): 67,
     (22, 25): 196,
     (22, 30): 21,
     (23, 5): 159,
     (23, 10): 64,
     (23, 15): 168,
     (23, 20): 200,
     (23, 25): 100,
     (23, 30): 106,
     (24, 5): 4,
     (24, 10): 190,
     (24, 15): 85,
     (24, 20): 169,
     (24, 25): 38,
     (24, 30): 104,
     (25, 5): 12,
     (25, 10): 172,
     (25, 15): 176,
     (25, 20): 104,
     (25, 25): 32,
     (25, 30): 171,
     (26, 5): 4,
     (26, 10): 79,
     (26, 15): 98,
     (26, 20): 85,
     (26, 25): 112,
     (26, 30): 92,
     (27, 5): 0,
     (27, 10): 54,
     (27, 15): 102,
     (27, 20): 100,
     (27, 25): 48,
     (27, 30): 176,
     (28, 5): 49,
     (28, 10): 182,
     (28, 15): 105,
     (28, 20): 165,
     (28, 25): 169,
     (28, 30): 184,
     (29, 5): 123,
     (29, 10): 17,
     (29, 15): 101,
     (29, 20): 196,
     (29, 25): 108,
     (29, 30): 108,
     (30, 5): 35,
     (30, 10): 20,
     (30, 15): 177,
     (30, 20): 159,
     (30, 25): 123,
     (30, 30): 183,
     (31, 5): 1,
     (31, 10): 164,
     (31, 15): 189,
     (31, 20): 102,
     (31, 25): 45,
     (31, 30): 112,
     (32, 5): 32,
     (32, 10): 101,
     (32, 15): 132,
     (32, 20): 56,
     (32, 25): 104,
     (32, 30): 46,
     (33, 5): 4,
     (33, 10): 80,
     (33, 15): 36,
     (33, 20): 68,
     (33, 25): 176,
     (33, 30): 187,
     (34, 5): 20,
     (34, 10): 80,
     (34, 15): 104,
     (34, 20): 104,
     (34, 25): 96,
     (34, 30): 104,
     (35, 5): 143,
     (35, 10): 119,
     (35, 15): 42,
     (35, 20): 47,
     (35, 25): 47,
     (35, 30): 113,
     (36, 5): 123,
     (36, 10): 159,
     (36, 15): 169,
     (36, 20): 181,
     (36, 25): 184,
     (36, 30): 177,
     (37, 5): 134,
     (37, 10): 70,
     (37, 15): 112,
     (37, 20): 175,
     (37, 25): 92,
     (37, 30): 23,
     (38, 5): 1,
     (38, 10): 70,
     (38, 15): 112,
     (38, 20): 51,
     (38, 25): 42,
     (38, 30): 23,
     (39, 5): 65,
     (39, 10): 153,
     (39, 15): 153,
     (39, 20): 170,
     (39, 25): 168,
     (39, 30): 168,
     (40, 5): 74,
     (40, 10): 153,
     (40, 15): 165,
     (40, 20): 98,
     (40, 25): 43,
     (40, 30): 112,
     (41, 5): 69,
     (41, 10): 42,
     (41, 15): 168,
     (41, 20): 21,
     (41, 25): 193,
     (41, 30): 193,
     (42, 5): 168,
     (42, 10): 178,
     (42, 15): 91,
     (42, 20): 93,
     (42, 25): 183,
     (42, 30): 91,
     (43, 5): 85,
     (43, 10): 159,
     (43, 15): 86,
     (43, 20): 107,
     (43, 25): 86,
     (43, 30): 91,
     (44, 5): 157,
     (44, 10): 3,
     (44, 15): 23,
     (44, 20): 154,
     (44, 25): 170,
     (44, 30): 168,
     (45, 5): 123,
     (45, 10): 177,
     (45, 15): 171,
     (45, 20): 169,
     (45, 25): 181,
     (45, 30): 179,
     (46, 5): 153,
     (46, 10): 181,
     (46, 15): 91,
     (46, 20): 171,
     (46, 25): 169,
     (46, 30): 184,
     (47, 5): 79,
     (47, 10): 89,
     (47, 15): 25,
     (47, 20): 165,
     (47, 25): 159,
     (47, 30): 159,
     (48, 5): 119,
     (48, 10): 54,
     (48, 15): 161,
     (48, 20): 27,
     (48, 25): 177,
     (48, 30): 173,
     (49, 5): 1,
     (49, 10): 61,
     (49, 15): 47,
     (49, 20): 92,
     (49, 25): 2,
     (49, 30): 28,
     (50, 5): 178,
     (50, 10): 164,
     (50, 15): 91,
     (50, 20): 10,
     (50, 25): 43,
     (50, 30): 196,
     (51, 5): 46,
     (51, 10): 171,
     (51, 15): 130,
     (51, 20): 111,
     (51, 25): 130,
     (51, 30): 131,
     (52, 5): 0,
     (52, 10): 183,
     (52, 15): 35,
     (52, 20): 168,
     (52, 25): 104,
     (52, 30): 159,
     (53, 5): 87,
     (53, 10): 12,
     (53, 15): 85,
     (53, 20): 107,
     (53, 25): 104,
     (53, 30): 177,
     (54, 5): 25,
     (54, 10): 113,
     (54, 15): 113,
     (54, 20): 23,
     (54, 25): 113,
     (54, 30): 92,
     (55, 5): 155,
     (55, 10): 155,
     (55, 15): 111,
     (55, 20): 32,
     (55, 25): 104,
     (55, 30): 34,
     (56, 5): 74,
     (56, 10): 181,
     (56, 15): 137,
     (56, 20): 74,
     (56, 25): 170,
     (56, 30): 159,
     (57, 5): 4,
     (57, 10): 168,
     (57, 15): 105,
     (57, 20): 102,
     (57, 25): 95,
     (57, 30): 105,
     (58, 5): 189,
     (58, 10): 181,
     (58, 15): 89,
     (58, 20): 172,
     (58, 25): 183,
     (58, 30): 96,
     (59, 5): 148,
     (59, 10): 171,
     (59, 15): 160,
     (59, 20): 60,
     (59, 25): 60,
     (59, 30): 32,
     (60, 5): 165,
     (60, 10): 157,
     (60, 15): 182,
     (60, 20): 182,
     (60, 25): 169,
     (60, 30): 170,
     (61, 5): 74,
     (61, 10): 171,
     (61, 15): 123,
     (61, 20): 104,
     (61, 25): 35,
     (61, 30): 34,
     (62, 5): 1,
     (62, 10): 197,
     (62, 15): 195,
     (62, 20): 166,
     (62, 25): 159,
     (62, 30): 123,
     (63, 5): 29,
     (63, 10): 56,
     (63, 15): 56,
     (63, 20): 104,
     (63, 25): 34,
     (63, 30): 131,
     (64, 5): 20,
     (64, 10): 189,
     (64, 15): 68,
     (64, 20): 32,
     (64, 25): 190,
     (64, 30): 104,
     (65, 5): 166,
     (65, 10): 105,
     (65, 15): 178,
     (65, 20): 104,
     (65, 25): 20,
     (65, 30): 189,
     (66, 5): 189,
     (66, 10): 54,
     (66, 15): 35,
     (66, 20): 134,
     (66, 25): 32,
     (66, 30): 131,
     (67, 5): 64,
     (67, 10): 112,
     (67, 15): 175,
     (67, 20): 123,
     (67, 25): 10,
     (67, 30): 193,
     (68, 5): 111,
     (68, 10): 91,
     (68, 15): 91,
     (68, 20): 165,
     (68, 25): 74,
     (68, 30): 170,
     (69, 5): 0,
     (69, 10): 166,
     (69, 15): 175,
     (69, 20): 197,
     (69, 25): 123,
     (69, 30): 159,
     (70, 5): 1,
     (70, 10): 42,
     (70, 15): 169,
     (70, 20): 91,
     (70, 25): 85,
     (70, 30): 91,
     (71, 5): 46,
     (71, 10): 165,
     (71, 15): 189,
     (71, 20): 46,
     (71, 25): 68,
     (71, 30): 104,
     (72, 5): 8,
     (72, 10): 20,
     (72, 15): 197,
     (72, 20): 93,
     (72, 25): 173,
     (72, 30): 100,
     (73, 5): 1,
     (73, 10): 176,
     (73, 15): 9,
     (73, 20): 168,
     (73, 25): 27,
     (73, 30): 28,
     (74, 5): 99,
     (74, 10): 85,
     (74, 15): 170,
     (74, 20): 100,
     (74, 25): 75,
     (74, 30): 107,
     (75, 5): 61,
     (75, 10): 2,
     (75, 15): 82,
     (75, 20): 42,
     (75, 25): 92,
     (75, 30): 3,
     (76, 5): 23,
     (76, 10): 25,
     (76, 15): 25,
     (76, 20): 169,
     (76, 25): 20,
     (76, 30): 24,
     (77, 5): 70,
     (77, 10): 168,
     (77, 15): 104,
     (77, 20): 190,
     (77, 25): 171,
     (77, 30): 46,
     (78, 5): 0,
     (78, 10): 149,
     (78, 15): 81,
     (78, 20): 60,
     (78, 25): 46,
     (78, 30): 176,
     (79, 5): 194,
     (79, 10): 68,
     (79, 15): 160,
     (79, 20): 26,
     (79, 25): 56,
     (79, 30): 60,
     (80, 5): 119,
     (80, 10): 165,
     (80, 15): 91,
     (80, 20): 20,
     (80, 25): 159,
     (80, 30): 169,
     (81, 5): 1,
     (81, 10): 153,
     (81, 15): 159,
     (81, 20): 26,
     (81, 25): 153,
     (81, 30): 158,
     (82, 5): 1,
     (82, 10): 123,
     (82, 15): 100,
     (82, 20): 183,
     (82, 25): 91,
     (82, 30): 166,
     (83, 5): 4,
     (83, 10): 177,
     (83, 15): 169,
     (83, 20): 159,
     (83, 25): 192,
     (83, 30): 158,
     (84, 5): 130,
     (84, 10): 38,
     (84, 15): 60,
     (84, 20): 132,
     (84, 25): 160,
     (84, 30): 171,
     (85, 5): 111,
     (85, 10): 32,
     (85, 15): 85,
     (85, 20): 26,
     (85, 25): 104,
     (85, 30): 189,
     (86, 5): 145,
     (86, 10): 130,
     (86, 15): 101,
     (86, 20): 104,
     (86, 25): 190,
     (86, 30): 32,
     (87, 5): 1,
     (87, 10): 160,
     (87, 15): 46,
     (87, 20): 188,
     (87, 25): 171,
     (87, 30): 32,
     (88, 5): 38,
     (88, 10): 155,
     (88, 15): 176,
     (88, 20): 199,
     (88, 25): 81,
     (88, 30): 81,
     (89, 5): 134,
     (89, 10): 200,
     (89, 15): 100,
     (89, 20): 170,
     (89, 25): 176,
     (89, 30): 129,
     (90, 5): 49,
     (90, 10): 98,
     (90, 15): 91,
     (90, 20): 15,
     (90, 25): 75,
     (90, 30): 105,
     (91, 5): 39,
     (91, 10): 74,
     (91, 15): 91,
     (91, 20): 91,
     (91, 25): 189,
     (91, 30): 169,
     (92, 5): 91,
     (92, 10): 166,
     (92, 15): 78,
     (92, 20): 47,
     (92, 25): 193,
     (92, 30): 175,
     (93, 5): 91,
     (93, 10): 85,
     (93, 15): 62,
     (93, 20): 25,
     (93, 25): 101,
     (93, 30): 170,
     (94, 5): 1,
     (94, 10): 42,
     (94, 15): 179,
     (94, 20): 169,
     (94, 25): 181,
     (94, 30): 181,
     (95, 5): 20,
     (95, 10): 154,
     (95, 15): 20,
     (95, 20): 169,
     (95, 25): 169,
     (95, 30): 173,
     (96, 5): 2,
     (96, 10): 3,
     (96, 15): 164,
     (96, 20): 175,
     (96, 25): 112,
     (96, 30): 47,
     (97, 5): 83,
     (97, 10): 170,
     (97, 15): 79,
     (97, 20): 42,
     (97, 25): 42,
     (97, 30): 112,
     (98, 5): 107,
     (98, 10): 47,
     (98, 15): 106,
     (98, 20): 175,
     (98, 25): 42,
     (98, 30): 123,
     (99, 5): 99,
     (99, 10): 123,
     (99, 15): 104,
     (99, 20): 176,
     (99, 25): 46,
     (99, 30): 187,
     (100, 5): 159,
     (100, 10): 166,
     (100, 15): 134,
     (100, 20): 45,
     (100, 25): 165,
     (100, 30): 112,
     (101, 5): 29,
     (101, 10): 153,
     (101, 15): 3,
     (101, 20): 141,
     (101, 25): 2,
     (101, 30): 159,
     (102, 5): 1,
     (102, 10): 169,
     (102, 15): 93,
     (102, 20): 104,
     (102, 25): 171,
     (102, 30): 170,
     (103, 5): 58,
     (103, 10): 153,
     (103, 15): 84,
     (103, 20): 185,
     (103, 25): 123,
     (103, 30): 165,
     (104, 5): 85,
     (104, 10): 164,
     (104, 15): 43,
     (104, 20): 112,
     (104, 25): 170,
     (104, 30): 164,
     (105, 5): 153,
     (105, 10): 179,
     (105, 15): 93,
     (105, 20): 185,
     (105, 25): 168,
     (105, 30): 169,
     (106, 5): 143,
     (106, 10): 20,
     (106, 15): 159,
     (106, 20): 94,
     (106, 25): 89,
     (106, 30): 166,
     (107, 5): 71,
     (107, 10): 39,
     (107, 15): 123,
     (107, 20): 47,
     (107, 25): 39,
     (107, 30): 112,
     (108, 5): 179,
     (108, 10): 159,
     (108, 15): 169,
     (108, 20): 169,
     (108, 25): 154,
     (108, 30): 159,
     (109, 5): 176,
     (109, 10): 46,
     (109, 15): 187,
     (109, 20): 187,
     (109, 25): 187,
     (109, 30): 48,
     (110, 5): 1,
     (110, 10): 1,
     (110, 15): 83,
     (110, 20): 170,
     (110, 25): 200,
     (110, 30): 180,
     (111, 5): 35,
     (111, 10): 85,
     (111, 15): 45,
     (111, 20): 2,
     (111, 25): 193,
     (111, 30): 112,
     (112, 5): 62,
     (112, 10): 65,
     (112, 15): 90,
     (112, 20): 101,
     (112, 25): 189,
     (112, 30): 170,
     (113, 5): 20,
     (113, 10): 177,
     (113, 15): 159,
     (113, 20): 91,
     (113, 25): 159,
     (113, 30): 196,
     (114, 5): 62,
     (114, 10): 153,
     (114, 15): 3,
     (114, 20): 91,
     (114, 25): 168,
     (114, 30): 89,
     (115, 5): 42,
     (115, 10): 42,
     (115, 15): 175,
     (115, 20): 45,
     (115, 25): 92,
     (115, 30): 47,
     (116, 5): 99,
     (116, 10): 8,
     (116, 15): 130,
     (116, 20): 104,
     (116, 25): 32,
     (116, 30): 171,
     (117, 5): 6,
     (117, 10): 2,
     (117, 15): 47,
     (117, 20): 112,
     (117, 25): 47,
     (117, 30): 16,
     (118, 5): 123,
     (118, 10): 166,
     (118, 15): 94,
     (118, 20): 89,
     (118, 25): 183,
     (118, 30): 196,
     (119, 5): 134,
     (119, 10): 123,
     (119, 15): 75,
     (119, 20): 93,
     (119, 25): 75,
     (119, 30): 94,
     (120, 5): 20,
     (120, 10): 100,
     (120, 15): 8,
     (120, 20): 159,
     (120, 25): 179,
     (120, 30): 89,
     (121, 5): 137,
     (121, 10): 153,
     (121, 15): 3,
     (121, 20): 166,
     (121, 25): 169,
     (121, 30): 169,
     (122, 5): 42,
     (122, 10): 137,
     (122, 15): 169,
     (122, 20): 184,
     (122, 25): 159,
     (122, 30): 3,
     (123, 5): 80,
     (123, 10): 46,
     (123, 15): 101,
     (123, 20): 91,
     (123, 25): 103,
     (123, 30): 168,
     (124, 5): 65,
     (124, 10): 181,
     (124, 15): 168,
     (124, 20): 101,
     (124, 25): 179,
     (124, 30): 89,
     (125, 5): 157,
     (125, 10): 172,
     (125, 15): 172,
     (125, 20): 172,
     (125, 25): 165,
     (125, 30): 166,
     (126, 5): 87,
     (126, 10): 95,
     (126, 15): 91,
     (126, 20): 85,
     (126, 25): 189,
     (126, 30): 170,
     (127, 5): 153,
     (127, 10): 74,
     (127, 15): 12,
     (127, 20): 169,
     (127, 25): 54,
     (127, 30): 104,
     (128, 5): 92,
     (128, 10): 181,
     (128, 15): 94,
     (128, 20): 168,
     (128, 25): 159,
     (128, 30): 184,
     (129, 5): 1,
     (129, 10): 164,
     (129, 15): 123,
     (129, 20): 2,
     (129, 25): 195,
     (129, 30): 123,
     (130, 5): 151,
     (130, 10): 168,
     (130, 15): 91,
     (130, 20): 196,
     (130, 25): 96,
     (130, 30): 169,
     (131, 5): 11,
     (131, 10): 173,
     (131, 15): 192,
     (131, 20): 4,
     (131, 25): 104,
     (131, 30): 12,
     (132, 5): 46,
     (132, 10): 159,
     (132, 15): 169,
     (132, 20): 169,
     (132, 25): 159,
     (132, 30): 165,
     (133, 5): 123,
     (133, 10): 181,
     (133, 15): 3,
     (133, 20): 23,
     (133, 25): 112,
     (133, 30): 28,
     (134, 5): 52,
     (134, 10): 134,
     (134, 15): 25,
     (134, 20): 23,
     (134, 25): 16,
     (134, 30): 112,
     (135, 5): 168,
     (135, 10): 62,
     (135, 15): 91,
     (135, 20): 100,
     (135, 25): 85,
     (135, 30): 107,
     (136, 5): 20,
     (136, 10): 165,
     (136, 15): 75,
     (136, 20): 172,
     (136, 25): 170,
     (136, 30): 91,
     (137, 5): 2,
     (137, 10): 160,
     (137, 15): 160,
     (137, 20): 104,
     (137, 25): 60,
     (137, 30): 42,
     (138, 5): 48,
     (138, 10): 176,
     (138, 15): 131,
     (138, 20): 48,
     (138, 25): 160,
     (138, 30): 34,
     (139, 5): 34,
     (139, 10): 46,
     (139, 15): 190,
     (139, 20): 189,
     (139, 25): 91,
     (139, 30): 43,
     (140, 5): 57,
     (140, 10): 134,
     (140, 15): 25,
     (140, 20): 57,
     (140, 25): 60,
     (140, 30): 104,
     (141, 5): 120,
     (141, 10): 70,
     (141, 15): 47,
     (141, 20): 43,
     (141, 25): 47,
     (141, 30): 10,
     (142, 5): 32,
     (142, 10): 20,
     (142, 15): 179,
     (142, 20): 170,
     (142, 25): 185,
     (142, 30): 180,
     (143, 5): 39,
     (143, 10): 29,
     (143, 15): 153,
     (143, 20): 91,
     (143, 25): 153,
     (143, 30): 169,
     (144, 5): 74,
     (144, 10): 165,
     (144, 15): 196,
     (144, 20): 168,
     (144, 25): 177,
     (144, 30): 91,
     (145, 5): 0,
     (145, 10): 134,
     (145, 15): 179,
     (145, 20): 154,
     (145, 25): 169,
     (145, 30): 123,
     (146, 5): 86,
     (146, 10): 41,
     (146, 15): 165,
     (146, 20): 196,
     (146, 25): 189,
     (146, 30): 172,
     (147, 5): 90,
     (147, 10): 97,
     (147, 15): 109,
     (147, 20): 108,
     (147, 25): 101,
     (147, 30): 166,
     (148, 5): 20,
     (148, 10): 85,
     (148, 15): 100,
     (148, 20): 100,
     (148, 25): 3,
     (148, 30): 93,
     (149, 5): 0,
     (149, 10): 168,
     (149, 15): 16,
     (149, 20): 42,
     (149, 25): 169,
     (149, 30): 164,
     (150, 5): 85,
     (150, 10): 75,
     (150, 15): 97,
     (150, 20): 85,
     (150, 25): 85,
     (150, 30): 75,
     (151, 5): 1,
     (151, 10): 134,
     (151, 15): 191,
     (151, 20): 193,
     (151, 25): 175,
     (151, 30): 26,
     (152, 5): 181,
     (152, 10): 97,
     (152, 15): 50,
     (152, 20): 170,
     (152, 25): 74,
     (152, 30): 179,
     (153, 5): 20,
     (153, 10): 123,
     (153, 15): 160,
     (153, 20): 104,
     (153, 25): 171,
     (153, 30): 54,
     (154, 5): 99,
     (154, 10): 153,
     (154, 15): 25,
     (154, 20): 186,
     (154, 25): 165,
     (154, 30): 169,
     (155, 5): 111,
     (155, 10): 130,
     (155, 15): 136,
     (155, 20): 160,
     (155, 25): 46,
     (155, 30): 161,
     (156, 5): 91,
     (156, 10): 169,
     (156, 15): 159,
     (156, 20): 165,
     (156, 25): 183,
     (156, 30): 8,
     (157, 5): 1,
     (157, 10): 95,
     (157, 15): 85,
     (157, 20): 183,
     (157, 25): 3,
     (157, 30): 92,
     (158, 5): 8,
     (158, 10): 85,
     (158, 15): 95,
     (158, 20): 159,
     (158, 25): 181,
     (158, 30): 170,
     (159, 5): 181,
     (159, 10): 85,
     (159, 15): 184,
     (159, 20): 184,
     (159, 25): 123,
     (159, 30): 166,
     (160, 5): 140,
     (160, 10): 43,
     (160, 15): 158,
     (160, 20): 159,
     (160, 25): 182,
     (160, 30): 2,
     (161, 5): 8,
     (161, 10): 85,
     (161, 15): 62,
     (161, 20): 170,
     (161, 25): 85,
     (161, 30): 62,
     (162, 5): 181,
     (162, 10): 169,
     (162, 15): 165,
     (162, 20): 182,
     (162, 25): 158,
     (162, 30): 166,
     (163, 5): 168,
     (163, 10): 54,
     (163, 15): 170,
     (163, 20): 36,
     (163, 25): 171,
     (163, 30): 131,
     (164, 5): 0,
     (164, 10): 32,
     (164, 15): 161,
     (164, 20): 177,
     (164, 25): 164,
     (164, 30): 170,
     (165, 5): 159,
     (165, 10): 168,
     (165, 15): 166,
     (165, 20): 166,
     (165, 25): 166,
     (165, 30): 168,
     (166, 5): 132,
     (166, 10): 43,
     (166, 15): 92,
     (166, 20): 175,
     ...}




```python
l2s_data = pd.DataFrame({"id_video" : [i for i in range(len(index_test))]})
for ts in train_sizes:
    l2s_data["conf"+str(ts)] = [l2s_confs[(i, ts)] for i in range(len(index_test))]
```


```python
l2s_data.set_index("id_video").to_csv("../../../results/raw_data/L2S_results.csv")
```


```python

```


```python

```


```python

```
