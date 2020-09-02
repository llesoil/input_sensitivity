# starts from ubuntu 18.04 version
FROM ubuntu:18.04
# update packages
RUN  apt-get update
# useful to get some files
RUN apt-get install -y wget
# compression tool (for files, not related to x264)
RUN apt-get install xz-utils
# install x264
RUN apt-get install -y x264=2:0.152.2854+gite9a5903-2
# install python
RUN apt-get install -y python3-pip
# install pandas
RUN pip3 install pandas
# create a directory for experiments go in there
RUN mkdir ~/experiments\
&& cd ~/experiments
# create dir for logs, dir for results, dir for scripts
RUN mkdir logs\
 && mkdir res\
 && mkdir scripts
# download the configurations
RUN wget https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/dataset_building/ugc/configs.csv
# a python script that generates sh scripts (one sh script per configuration)
RUN wget https://anonymous.4open.science/repository/df319578-8767-47b0-919d-a8e57eb67d25/replication/dataset_building/ugc/generate_sh.py
# run it
RUN python3 generate_sh.py
