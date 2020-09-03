# Learning Once and for All ?
# On the Input Sensitivity of Configurable Systems

Dear ICSE reviewers,

In this repository, you can consult the code, the data and the results related to our submission "Learning Once and for All? On the Input Sensitivity of Configurable Systems".

# GOAL

In this paper, we want to prove that **input videos have an influence on x264 compression performances**.

Once this statement is proved, we use the properties of input videos (e.g. height, width, spatial and temporal complexities, etc.) to **find a configuration optimizing a performance** (in our case minimize the bitrate of video compressions).


# Material

## Introduction

As an introduction, let us separate two different parts of the experimental protocol;

- **Measurements** - first of all, we measure software performances, for a given set of 201 configurations and for a dataset of 1397 videos.

- **Analyze** - we then analyze the obtained measurements with python scripts.


### I- Measurements

## I- 1. Replication

We provide a docker image to build (i.e. a Dockerfile) to reproduce the measurements (i.e. see these [files](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/data/ugc/res_ugc/)).

To use it, you can follow the steps in this directory:

https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/dataset_building/ugc/

See README.md in the replication folder.

#### I- 2. Data

See the README.md file in the data folder.

In this experiment, we consider two types of data; 

##### I- 2. a-] Encoded videos

We use the dataset of videos defined in "Youtube UGC Dataset for Video Compression Research" by Wang et al.

These videos are available in the Youtube UGC Dataset cloud:

https://console.cloud.google.com/storage/browser/ugc-dataset/original_videos;tab=objects?prefix=

Alternatively, to see the content of each video, you can check 'Explore' in the Youtube UGC Dataset homepage:

https://media.withyoutube.com/

You can check the data folder to see the list of videos used for this project.

##### I- 2. b-] Measurements

All the measurements are in the data/res_ugc [folder](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/data/ugc/res_ugc/).

Data is also available here: https://zenodo.org/record/3928253


### II- Analyze

#### II- 1. Code

All the code is in the src folder. To run this code, you will have 

We highly recommend to consult this page, containing code and additional explanations on the paper results; 

https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/src/main/bitrate.md



#### II- 2. Resources

Additional information about the experiment, the x264 configuration knowledge (a.k.a. domain knowledge), and ideas explored during the redaction of the paper!

See README.md in the resources folder.



#### II- 3. Results

Results produced by the code, previously presented in the src folder.

See README.md in the results folder.



# Acknowledgement

We would like to thank Yilin Wang for the explanations about input properties.


