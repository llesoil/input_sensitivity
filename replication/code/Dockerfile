# starts from ubuntu 18.04 version
FROM ubuntu:18.04
# update packages
RUN apt-get update
# install git 
RUN apt-get install -y git
# install python and python libraries
RUN apt-get install -y python3-pip\ 
 && pip3 install pandas scikit-learn seaborn
# download the code
RUN git clone https://github.com/anonymous4opens/code
# create dir for results
RUN cd code\
 && mkdir results
# && cd src/main \
# && python3 bitrate.py
