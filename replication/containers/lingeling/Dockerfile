FROM ubuntu:20.04
# update packages
RUN apt -y update
# install git, gcc, cmake, and libraries dedicated to compilation
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git make automake gcc cmake autopoint libtool m4 automake po4a lzma nano liblzma-dev g++ flex
# clone lingeling
#RUN git clone https://github.com/arminbiere/lingeling \
# && cd gingeling \
# && bash ./configure.sh
# && make
# copy files
COPY . /
