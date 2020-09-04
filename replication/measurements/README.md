We provide you some instructions, to reproduce the measurements in the same environment :


## 1. Install Docker

If docker is not installed on your laptop, just follow this link; https://docs.docker.com/get-docker/

When it's done, you should be able to run the following command : 

```docker --version```

    >```Docker version 19.03.12, build 48a66213fe```


Useful links if you struggle with docker installation :

- https://www.ibm.com/cloud/blog/installing-docker-windows-fixes-common-problems

- https://medium.com/@randima.somathilaka/docker-engine-install-fails-in-ubuntu-3e70762c2187

- https://runnable.com/docker/install-docker-on-macos


## 2. Build the image of the container

Check that you are in the root/replication/measurements/ folder, and run the following command;

```sudo docker build -t inputs .```

You will need administrator privileges to run the command. It will build a docker image with the x264 version in a 18.04 linux environment (~800MB of space).

You can skip the following lines of information and go straight to 3.

Additionally, you can open the Dockerfile in this directory to see the different steps of the building.

- First, we use a linux 18.04 environment, we update the packages

- We install git to download code (from an "anonymous account")

- We install the x264 version we used for our experiment (version=2:0.152.2854+gite9a5903-2)

- We install python (version 3) and the pandas library

- We clone a git repository (anonymous account, you can check the repo) to add scripts (e.g. a python script that generates sh scripts) and the configuration csv

- We create a directory "logs" for logs, a directory "scripts" for scripts (one sh script per configuration) and a directory "res" for results. We add a directory for videos and provide two videos extracted from the Youtube UGC Dataset (but compressed to gain some space in the github repository) to test it. The bash script listFile.sh lists the different videos in the videos folder, put their names and extensions in the listVideos.csv file. The generate_sh.py generates sh scripts. The launchUGC.sh launches the measurements.
Once downloaded, you can launch the image by typing the following command line


## 3. Launch measurement

To run the docker image, just type:

```sudo docker run -it inputs```

This command will require to be in the docker container, in interactive mode. 

Go to the  directory

```cd code```

Launch the measurements

```bash launchUGC.sh```

The measurements are done on the videos that are in the listVideo.csv file (here Animation_360P-24d4.mp4 and Lecture_360P-03bc.mp4).

For each video, we compute the 201 configurations, and store it in a csv file in the res directory.

For an example, Lecture_360P-03bc.mp4 will be compressed 201 times by x264; once these compressions are finished, a Lecture_360P-03bc.csv file appears in the res folder. 


### To add videos

- Download the videos in the videos directory:

- Run the listFile.sh script, i.e. ```bash listFile.sh```

- Launch the measurements, i.e. ```bash launchUGC.sh```

- Check the res directory for results when it is done

### See the repository for the files ; https://github.com/anonymous4opens/experiment


 
