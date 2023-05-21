#FROM amd64/ubuntu:22.04
FROM ubuntu:lunar-20230415

ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV MTC_PATH=/opt/modulus_mtc
ENV PATH=${MTC_PATH}/bin:${PATH}
ENV MPC_PATH=${MTC_PATH}/mpc
ENV PYTHONPATH=${MTC_PATH}:.

RUN apt update \
    && apt install wget libgl1-mesa-glx libxrender1 graphviz git  -y \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh \
    && sh Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -b -p /usr/local/miniconda3 \
    && /usr/local/miniconda3/bin/conda init bash  \
    && rm Miniconda3-py38_23.3.1-0-Linux-x86_64.sh \
    && apt remove -y wget \
    && apt-get autoclean \
    && apt-get autoremove -y --allow-remove-essential \
    && rm -rf /var/lib/apt/lists/* 

COPY . ${MTC_PATH}

RUN cd ${MTC_PATH} \
    && . /root/.bashrc \
    && conda create -n mtcenv python=3.8.16 -y\
    && conda activate mtcenv \
    && echo "conda activate mtcenv" >> /root/.bashrc \
    && pip install -r requirements.txt \
    && rm -rf ~/.cache/pip 

RUN mkdir /workspace

WORKDIR /workspace

# ---------------------------------------
# Create unprivileged user
# ---------------------------------------

# ARG MTC_USER="pmler"
# ARG MTC_UID="1000"
# ARG MTC_GID="1000"

# SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# USER root

# ENV SHELL=/bin/bash \
#     MTC_USER="${MTC_USER}" \
#     MTC_UID=${MTC_UID} \
#     MTC_GID=${MTC_GID}

# RUN useradd -l -m -s /bin/bash -N -u "${MTC_UID}" "${MTC_USER}" && \
#     chmod g+w /etc/passwd   

# USER ${MTC_UID}

# RUN /usr/local/miniconda3/bin/conda init bash && \
#     echo "conda activate mtcenv" >> /home/${MTC_USER}/.bashrc 

# WORKDIR /home/$MTC_USER

#---------------------------------------------------------
# Done
#---------------------------------------------------------
# RUN apt-get autoclean \
#  && apt-get autoremove -y --allow-remove-essential \
#  && rm -rf /var/lib/apt/lists/* \
#  && rm -rf ~/.cache/pip

