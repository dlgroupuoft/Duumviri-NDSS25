FROM ubuntu:20.04
USER root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app
COPY . /app
RUN cd /app


RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        curl git-all build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev wget \
        libncurses5-dev libgdbm-dev libnss3-dev libgdm-dev \
        python3.8 python3.8-venv python3.8-dev python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip


RUN ./build/install_packages.sh 
RUN cd /app
RUN ./build/apply_package_modifications.sh /usr/local \
    && pip uninstall -y keras \
    && pip uninstall -y tensorflow \
    && pip install tensorflow==2.11.0 \
    && pip uninstall -y tensorflow-hub \
    && pip install tensorflow-hub==0.16.0 
    

# Installsl all necessary componenets
RUN apt-get update && apt-get install -y --no-install-recommends lzma sqlite3 ffmpeg libsm6 libxext6 libgl1 libenchant1c2a libatk1.0-0 libatk-bridge2.0-0 libcups2 libxcomposite-dev libxdamage1 libgbm-dev xvfb nodejs npm python-setuptools python3-distutils  nano libnss3-dev pkg-config lsb-release sudo kmod make wget ca-certificates llvm libncurses5-dev mecab-ipadic-utf8 python-openssl libfuse2

# start dbus
RUN dbus-uuidgen > /etc/machine-id
RUN service dbus start

# install npm and node
RUN npm install npm@latest -g && \
    npm install n -g && \
    n latest 
RUN hash -r && node -v && npm -v


# install pagegraph-crawl
RUN cd code \
&& rm -rf pagegraph-crawl \
&& git clone https://github.com/brave/pagegraph-crawl.git  \
&& cd pagegraph-crawl \
&& git checkout aab1e3eb339febf7a05f68c744498687cd07a61e \
&& npm install \
&& npm run build \
&& cp ../run.sh ./ \
&& mkdir -p test_output && ./run.sh https://homedepot.com test_output/ && ls test_output/

RUN cd /app
CMD ["/bin/bash"]
