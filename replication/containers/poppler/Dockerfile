FROM ubuntu:20.04
# update packages
RUN apt -y clean
RUN apt -y update
# install git, gcc, cmake, and libraries dedicated to compilation
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git make automake gcc cmake libopenjp2-7 build-essential libfreetype6 libfreetype6-dev pkg-config libfontconfig1-dev libopenjp2-7-dev libjpeg-dev curl libcairo2-dev libgirepository1.0-dev nano libcurl4-openssl-dev libgtk-3-dev libgtk2.0-dev libboost-system-dev libboost-filesystem-dev libboost-locale-dev libboost-iostreams-dev libnss3-tools libtiff-dev
# clone poppler
RUN git clone https://github.com/freedesktop/poppler \
 && cd poppler \
 && git reset --hard 42dde68
# First compilation with the default configuration
RUN cd poppler \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make
# copy files
COPY . /

