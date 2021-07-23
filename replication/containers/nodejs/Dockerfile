FROM fedora:33
# update packages
RUN dnf -y up --exclude=kernel --exclude=kernel-core --exclude=kernel-devel --exclude=kernel-headers --exclude=python --exclude=gcc-c++ --exclude=make
# https://github.com/nodejs/node/blob/master/BUILDING.md
RUN dnf install -y python gcc-c++ make wget nano
RUN uname -r >> infos \
 && gcc --version >> infos \
 && make -v >> infos \
 && python --version >> infos
# clone nodejs
RUN wget https://github.com/nodejs/node/archive/refs/tags/v15.14.0.tar.gz \
 && tar -xf v15.14.0.tar.gz
# command to compile node with the default configurations
RUN cd node-15.14.0 \
 && ./configure \
 && make -j4
# Add files to run the scripts
COPY . /

 
