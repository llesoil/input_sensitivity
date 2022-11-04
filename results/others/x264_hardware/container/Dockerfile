# starts from ubuntu 18.04 version
FROM ubuntu:18.04
# update packages
RUN apt-get update
# install git 
RUN apt-get install -y git 
RUN apt-get install -y sshpass
# install x264, the version we used for the available measurements
RUN apt-get install -y x264=2:0.152.2854+gite9a5903-2
# note : this x264 version is deprecated for ubuntu 20.04
# copy the files
COPY . /
# create dir for logs, dir for results, dir for scripts, generate sh scripts, dl videos
RUN mkdir output \
 && bash listFile.sh
CMD bash launchMeasures.sh
