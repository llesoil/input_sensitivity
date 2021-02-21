# Learning algorithm per approach

## *Simple Learning*

### Best Compromise

**Best compromise (BC)** applies a performance model on all the training set, without making a difference between input videos. 
It selects the configuration working best for most videos in the training set. 
Technically, we rank the 201 configurations (1 being the optimal configuration, and 201 the worst) and select the one optimizing the sum of ranks for all input videos in the training set. 

Best algorithm : Random Forest

HP : {'max_depth': 5, 'max_features': 15, 'min_samples_leaf': 5, 'n_estimators': 100}

### Model Reuse

We arbitrarily choose a first video, learn a performance model on it, and select the best configuration minimizing the performance for this video. This approach represents the error made by a model trained on a  source input (i.e., a  first video) and transposed to a target input (i.e., a second video, different from the first one), without considering the difference of content between the source and the target. In theory, it corresponds to a fixed configuration, optimized for the first video. We add Model Reuse as a witness approach to measure how we can improve the standard performance model.

The **Model Reuse** selects a video of the training set, apply a model on it and keep a near-optimal configuration working for this video. Then, it applies this configuration to all inputs of the test set.

Best algorithm : Random Forest Regressor
Close to Gradient Boosting Trees and Decision Trees

HP : {'max_depth' : None, 'max_features' : 33, 'min_sample_leaf' : 2, 'n_estimators' : 50}

## *Learning with Properties*

These learning approaches incorporate input properties in the learning phase to differentiate inputs during the training of the model. Input properties are features giving information about the content of input, e.g. the number of lines of code for a program fed to a compiler, or the resolution for a video fed to x264. We assume that these properties are available both for the training set and the test set of videos. 

### Input-aware Learning

**Input-aware Learning (IaL)** was first designed to overcome the input sensitivity of programs when compiling them with PetaBricks. 

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

Chosen algorithm Step 5: Gradient Boosting Trees
Chosen algorithm Step 6: Neural Network for all the groups


### Direct Inclusion


**Direct Inclusion (DI)** includes input properties directly in the model during the training phase. The trained model then predicts the performance of x264 based on a set of properties (i.e. information about the input video) **and** a set of configuration options (i.e. information about the configuration). We fed this model with the 201 configurations of our dataset, and the properties of the test videos. We select the configuration giving the best prediction (e.g. the lowest bitrate).

Chosen algorithm : Random Forest Regressor

HP : {'max_depth': None, 'max_features': 33, 'min_samples_leaf': 2, 'n_estimators': 100}

## *Transfer learning*

*Transfer learning* approaches reuse the measurements and the model trained on a **source** video to increase the accuracy obtained on a **target** video when predicting its optimal configuration. In our case, source videos are extracted from the training set, while the target is part of the test set. Unlike Input-aware Learning, these approaches only need to measure few configurations for each video of the test set.

For transfer learning, the chosen hyperparameter are sensible to the source and the target, so it seems more complicated to tune their hyperparameters.

### Beetle

**Beetle** is a transfer learning approach defined by Krishna et al that relies on *source selection*. 
Given a (set of) input(s), the goal is to rank the sources by performance, in order to discover a bellwether input from which we can easily transfer performances (i.e. find the best source). 
Then, we transfer performances from this bellwether input to all inputs of the test set. 
We only apply the discovery phase (i.e. the search of bellwether) on the training set, to avoid introducing any bias in the results. 

Chosen algorithm : Random Forest Regressor.

Seems to be buggy with SVR, always returns the same configuration, but the MSE is lower than with RFs. See the source code of SVR?


### Learning to Sample

**Learning to Sample (L2S)** is a transfer learning approach defined by Jamshidi et al. 
First, it exploits the source input and selects configurations that leverage influential (interactions of) features for this input. 
Then, it explores the similarities between the source and the target, thus adding configurations having similar performances for the source and the target. 
Finally, it uses the configurations selected in previous steps to efficiently train a model on the target input. 

Chosen algorithm : Random Forest Regressor

Linear regression works well (due to the construction of the algorithm), so we chose RF to avoid any bias.


### Model Shift

**Model Shift (MS)** is a transfer learning defined by Valov et al. 
First, it trains a performance model on the source input and predicts the performance distribution of the source input. 
Then, it trains a shifting function, predicting the performances of the target input based on the performances of the source. 
Finally, it applies the shifting function to the predictions of the source. 

Learning algorithm : Support Vector Regressor
Shifting Function : Random Forest Regressor


### No Transfer

**No Transfer (NT)** is a Simple Learning approach, acting as a control approach to state whether transfer learning is suited to solve this problem. 
It trains a performance model directly on the target input, without using any source. 
We expect to outperform No Transfer with transfer learning approaches. 

Chosen algorithm : Decision Tree Regressor

