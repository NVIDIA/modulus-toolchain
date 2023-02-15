# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# time sudo docker build -t mtc:$MTC_VERSION .
# To create tarball: time sudo docker save mtc:$MTC_VERSION | gzip > mtc_$MTC_VERSION.tar.gz
# To update remote: docker push [nvcr.io/]remote-location/mtc:$MTC_VERSION
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/modulus/modulus:22.07
FROM ${FROM_IMAGE_NAME}
# RUN pip install modulus==22.7 click==8.0.4 warp-lang==0.6.2 usd-core==22.11
COPY . /opt/modulus_mtc
# RUN cd /opt/modulus_mtc/mpc && \
#     pip install -r requirements-mtc-lab.txt
RUN cd /opt/modulus_mtc/mpc && \
  bash set-up-mtc-lab.sh && \
  cd /opt/modulus_mtc && \
  source set-up-env.sh 
    # pip list | awk '{print$1"=="$2}' | tail +3 > constraints.txt && \
    # cat requirements-mtc-lab.txt | xargs -i \
    #   pip install \
    #     --upgrade --upgrade-strategy only-if-needed \
    #     -c constraints.txt {} ||:
ENV PATH=/opt/modulus_mtc/bin:$PATH \
    PYTHONPATH=/opt/modulus_mtc \
    MPC_PATH=/opt/modulus_mtc/mpc \
    MTC_PATH=/opt/modulus_mtc
    # MTC_VERSION=`cat /opt/modulus_mtc/MTC_VERSION`
# Delete jupytext content manager override.
# RUN sed -i \
#       '/"jupytext.TextFileContentsManager"/d' \
#       /opt/conda/etc/jupyter/jupyter_notebook_config.py
# # Final JupyterLab Build
# # jupyter lab build --dev-build=False --minimize=False && \
# RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
#     apt update && apt install -y --no-install-recommends nodejs && \
#     jupyter lab build && \
#     jupyter lab clean && \
#     /opt/modulus_mtc/mpc/venv-mtc-lab/bin/jupyter lab build && \
#     /opt/modulus_mtc/mpc/venv-mtc-lab/bin/jupyter lab clean

# if based on Modulus:22.07 then need to patch the geometry module
RUN cp /opt/modulus_mtc/modulus2207-patch/geometry.py /modulus/modulus/geometry
RUN pip install warp-lang==0.6.2 usd-core==22.11 k3d==2.15.2 && \
    /opt/modulus_mtc/mpc/venv-mtc-lab/bin/pip install h5py
