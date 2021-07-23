FROM ubuntu:20.04
# update packages
RUN apt -y update
# install git, gcc, cmake, and libraries dedicated to compilation
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git make automake gcc cmake autopoint libtool m4 automake po4a lzma nano liblzma-dev
# clone xz
RUN git clone https://git.tukaani.org/xz.git \
 && cd xz \
 && git reset --hard e7da44d \
 && ./autogen.sh \
 && ./configure \
 && make
# copy files
COPY . /
