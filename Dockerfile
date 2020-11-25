FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter

RUN apt-get update && apt-get install wget
#RUN cd ~/Downloads
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
RUN $HOME/miniconda/bin/conda init
RUN source ~/.bashrc
ENV PATH /root/miniconda/bin:$PATH

RUN apt-get install -y vim-tiny

RUN conda update --all
RUN conda install ipykernel
RUN conda install mkl>=2018
RUN conda create -y -n pytorch3d python=3.8
RUN echo "conda activate pytorch3d" >> ~/.bashrc 

#RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch --yes
RUN conda install -c pytorch pytorch==1.6.0 torchvision cudatoolkit=10.2 --yes

#RUN conda install -c conda-forge -c takatosp1 fvcore --yes
RUN conda install -c conda-forge -c fvcore fvcore --yes

#RUN conda install -c cub 

#RUN conda install pytorch3d -c pytorch3d --yes
RUN conda install pytorch3d -c pytorch3d --yes

RUN pip install --upgrade pip 
RUN pip install scikit-image matplotlib imageio plotly opencv-python
RUN pip install pip install black 'isort<5' flake8 flake8-bugbear flake8-comprehensions

#RUN pip install --upgrade pip && \
#    /bin/bash -c ". activate pytorch3d && \
#    pip install --upgrade pip && \
#    pip install \
#    scikit-image \
#    matplotlib \
#    imageio \
#    black \
#    tqdm \
#    pandas \
#    scipy \
#    opencv-python \
#    isort \
#    flake8 \
#    flake8-bugbear \
#    flake8-comprehensions"

#RUN conda install --name pytorch3d pytorch3d -c pytorch3d --yes
