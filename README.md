# Learning Once and for All ?
# On the Input Sensitivity of Configurable Systems

Dear ICSE reviewers,

In this repository, you can consult the code, the data and the results related to our submission "Learning Once and for All? On the Input Sensitivity of Configurable Systems".



# GOAL

In this paper, we want to prove that **input videos have an influence on x264 compression performances**.

In particular, the first research question is :
RQ1. Do Input Videos Change Performances of x264 Configurations?

We aim at quantifying the differences of x264's performances due to input videos, and how input videos interact with software features.

Once this statement is proved, we use the properties of input videos (e.g. height, width, spatial and temporal complexities, etc.) to **find a configuration optimizing a performance** (in our case minimize the bitrate of video compressions).

We propose an approach, Inputec, that finds a configuration tailored for inputs.

The second research question is :
RQ2. Can we use Inputec to find configurations adapted to input videos?

We detail the results of our experiment in the related paper.

To fully understand the code and results, we advise to read the related submission paper.



# Material

## Introduction

As an introduction, let us separate two different parts of the experimental protocol;

- **Measurements** - first of all, we measure x264 performances, for a given set of 201 configurations and a dataset of 1397 videos.

- **Analyze** - we then analyze the obtained measurements with python scripts.



### I- Measurements

The third contribution of the paper states that for replicability, we provide a comprehensive **data**set of configurations’ measurements. 

We provide instructions useful to **replicate** or reproduce our dataset.

#### I- 1. Data

Data are available in the "data" folder.

For this experiment, showing that input videos have an influence on video compression's performances, we first need **videos** to encode.

##### I- 1. a-] Encoded videos

We used the dataset of videos defined in "Youtube UGC Dataset for Video Compression Research" by Wang et al. (see their [publication](https://arxiv.org/abs/1904.06457)).

Since it is a huge dataset of videos (about 3 terabytes of videos), we did not duplicate it.

However, these videos are available in the [Youtube UGC Dataset cloud](https://console.cloud.google.com/storage/browser/ugc-dataset/original_videos;tab=objects?prefix=)

Alternatively, and to see more concretely the content of each video, you can visualize them by clicking on 'Explore' in the Youtube UGC Dataset homepage:

https://media.withyoutube.com/

For each of these videos, and for 201 configurations, we compressed the video with x264 and measured five performances (bitrate, cpu consumption, number of frames encoded per second, encoded size of videos and encoding time). For each video, there is a related .csv file that contains these **measurements**.

##### I- 1. b-] Measurements

All these measurements are in the data/ugc/res_ugc folder, follow this [link](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/data/ugc/res_ugc/).

We shared our measurements in zenodo.org, in order to make it accessible for everyone: https://zenodo.org/record/3928253

You may be interested to **replicate** the measurements with other datasets of videos.

#### I- 2. Replication

We provide a **docker image to build (i.e. a Dockerfile) to replicate or reproduce the measurements**.

To use it, you can follow the steps in the "replication/measurements" directory:

https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/measurements

The "replication" folder details the configuration of our server (hardware details, operating system distribution and version, software version) and contains a Dockerfile to build in order to replicate the measurements.

Data about the experiment, x264 configuration knowledge (a.k.a. domain knowledge), and ideas explored during the redaction of the paper are available in the "information" sub-directory.



### II- Analyze

To analyze these measurements, we use **code**, that gives **results**, depicted and discussed in the submission paper.

#### II- 1. Code

The code is in the "src" folder.

**We highly recommend to consult this page, containing a preview of the (heavily commented) code leading to the results of the paper**; 

https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/src/main/bitrate.md

If you want to run it, we provide a runnable [python file](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/src/main/bitrate.py) generating these results and all the graphs of the paper.

This python file will require a working python environment and python libraires; see the [requirements.txt](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/code/requirements.txt) file.

We provide a [docker file](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/code/), build it to run the code and get the results!


#### II- 2. Results

Results produced by the code, previously presented in subsection II-1.

The ["results" folder](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/results) contains all the figures depicted in the paper.

Other results are in the sub-directory [others](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/results/others).


# Artifact evaluation

If you don't care about paper details, just type the following commands:

```git clone https://github.com/anonymous4opens/replication```

If a git error occurs, install git with the following line (linux):

```sudo apt-get install -y git```

Go in the replication directory:

```cd replication```

We provide:

1. a dockerfile to replicate measurements : the sub-directory [measurements](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/measurements/) contains a docker file, build it to reproduce the measurements. Go in the directory:

```cd measurements```

And follow the instructions of the README.md file.
When you are done with the measurements, exit the container:

```exit```

And go back to replication directory:

```cd ..```

2. a dockerfile to run the code : the sub-directory [code](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/code/) conatins a docker file, build it to run the code and get the results.

Go in the directory:

```cd code```

And follow the instructions of the README.md file.
When you are done with the measurements, exit the container:

```exit```

Don't forget to remove the two docker containers (```sudo docker ps -all```, and ```sudo docker rm [replace with the container ids]```)

Thanks for testing our artifact!

# Acknowledgement

We would like to thank Yilin Wang for the explanations about input properties.


