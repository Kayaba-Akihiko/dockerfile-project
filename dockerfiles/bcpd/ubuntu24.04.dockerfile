#  Copyright (c) 2026. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

# sudo docker buildx build --platform linux/amd64,linux/arm64 --tag akihikokayaba/bcpd:ubuntu24.04 --push . 2>&1 | tee /win/flounder/user/koku/temp/build.log

FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        wget \
        unzip \
        make \
        libomp-dev \
        liblapack-dev \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/bcpd && \
    cd /opt/bcpd && \
    wget -q https://github.com/ohirose/bcpd/archive/refs/heads/master.zip && \
    unzip master.zip && \
    rm master.zip && \
    cd bcpd-master && \
    make -j"$(nproc)" OPT=-DUSE_OPENMP ENV=LINUX


FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libopenblas0 \
        liblapack3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/bcpd/bcpd-master /opt/bcpd

ENV PATH="/opt/bcpd:${PATH}"

ENTRYPOINT ["bcpd"]