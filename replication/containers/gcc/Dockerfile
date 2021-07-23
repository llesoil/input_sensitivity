FROM ubuntu:20.04
# update packages
RUN apt -y update
# install git, gcc, cmake, and libraries dedicated to compilation
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential git make automake gcc cmake autopoint libtool m4 automake po4a lzma nano liblzma-dev g++ flex
# clone gcc
RUN git clone https://github.com/gcc-mirror/gcc \
 && cd gcc \
 && ./contrib/download_prerequisites
RUN cd gcc \
 && ./configure --disable-multilib \
 && make bootstrap \
 && make install
# copy files
COPY . /
