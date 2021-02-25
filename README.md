# Predicting Performances of Configurable Systems
# The Issue of Input Sensitivity

Dear ESEC/FSE reviewers,

In this repository, you can consult the code, the data and the results related to our submission "Predicting Performances of Configurable Systems: the Issue of Input Sensitivity".

# Artifact evaluation

If you don't care about paper details, just follow the instructions:

- Install [docker](https://docs.docker.com/get-docker/). You can check your docker version (use the command line ```sudo docker --version```) and status (use ```sudo systemctl status docker```).

- Pull our image, by typing the following line in a terminal

```sudo docker pull anonymicse2021/fse21:latest```

- Run it in interactive mode

```sudo docker run -it anonymicse2021/fse21:latest```

You need to test the *measurement process* (i.e. validate that you can measure x264 performances with other videos) and the *code* (i.e. run our code on your laptop) :

## 1. Measurements

Go in the experiment folder:

```cd experiment```

Launch the measurements:

```bash launchUGC.sh```

The measurement process should take about 10 minutes, you can follow the executed command lines in the console. Two videos (in the videos directory) are processed; Animation_360P-24d4.mp4 and Lecture_360P-03bc.mp4

For each video, we compute the 201 configurations, and store the results of the compression (i.e. the performances, like the encoding bitrate, time or cpu) in a csv file in the res directory.

For an example, Lecture_360P-03bc.mp4 will be compressed 201 times by x264. Once these compressions are finished, a Lecture_360P-03bc.csv file appears in the res folder. Each line of this file contains a configuration of x264, and a set of performances (cpu, encoded size of output video, fps, average bitrate of compression, and compression duration).

## 2. Code

Then, go in the script directory :

```cd ../code/src/main```

And launch the python script :

```python3 bitrate.py```

In the command line outputs, you will see the outputs (as in the [outputs.txt](https://anonymous.4open.science/r/df319578-8767-47b0-919d-a8e57eb67d25/replication/code/outputs.txt) file) printed and explained in the related [markdown](https://anonymous.4open.science/r/df319578-8767-47b0-919d-a8e57eb67d25/src/main/bitrate.md).

The python script is relatively long to run entirely (about 30 minutes). Once the run is finished, you can attest that the results directory contains the figure printed in our paper:

```ls ../../results```

We apply modifications to corrmatrix-ugc-dendo-Spearman-kbs.pdf (i.e. convert it to png, add explanations about the scale and highlight groups), but the content (correlogram and dendograms) is the same as corrmatrix_modif.png (i.e. figure 1).

Figure 3 is just an overview of the state-of-the-art comparison, and is not an output of the code. The same applies to Figure 6.

You're done with the replication process, you can ```exit``` the container. 

Don't forget to remove the docker container (```sudo docker ps -all```, and ```sudo docker rm [replace with the container id]```)

Thanks for testing our artifact!


# GOAL

In this paper, we want to prove that **input videos have an influence on x264 compression performances**.

In particular, the first research question is :
RQ1. Do Input Videos Change Performances of x264 Configurations?

We aim at quantifying the differences of x264's performances due to input videos, and how input videos interact with software features.

Once this statement is proved, we use the properties of an input video (e.g. height, width, spatial and temporal complexity, etc.) to **find a configuration optimizing a performance for this video** (in our case minimize the bitrate of the video compression).

We propose to compare several state-of-the-art approaches finding a configuration tailored for inputs.

The second research question is:
RQ2. What could be a good tradeoff between the prediction and the cost of the prediction?

We detail the results of our experiment in the related paper.

To fully understand the code and results, we advise to read the related paper.



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

We provide a **docker image to build (i.e. a Dockerfile) to replicate or reproduce the measurements** (see the Artifact Evaluation above).

The "replication" directory details the configuration of our server (hardware details, operating system distribution and version, software version) and contains a Dockerfile (i.e a docker image to build in order to replicate the measurements).

Data about the experiment, x264 configuration knowledge (a.k.a. domain knowledge), and ideas explored during the redaction of the paper are available in the "information" sub-directory.



### II- Analyze

To analyze these measurements, we use **code**, that gives **results**, depicted and discussed in the paper attached to the submission.

#### II- 1. Code

The code is in the "src" folder.

**We highly recommend to consult this page, containing a preview of the (heavily commented) code leading to the results of the paper**; 

https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/src/main/bitrate.md

If you want to run it, we provide a runnable [python file](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/src/main/bitrate.py) generating these results and all the graphs of the paper.

This python file will require a working python environment and python libraires; see the [requirements.txt](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/code/requirements.txt) file.

We provide a [docker file](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/code/), build it to run the code and get the results! Or you can use our docker image, see the Artifact evaluation part above.


#### II- 2. Results

Results produced by the code, previously presented in subsection II-1.

The ["results" folder](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/results) contains all the figures depicted in the paper.

Other results are in the sub-directory [others](https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/results/others).


# Acknowledgement

We would like to thank Yilin Wang for his explanations about input properties.


