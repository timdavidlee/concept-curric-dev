#! /bin/bash

set -e

if [ $# -ne 1 ]; then
    echo "Usage $(basename $0) <target-host>"
    exit 1
fi


REMOTE_HOST=$1
# because aws ami, if debian / centos, the user will be different
USER='ec2-user'


echo "============ [ installing the yum stuff] ============"
ssh -T $USER@$REMOTE_HOST <<EOF
sudo yum update -y
sudo yum install -y \
    htop \
    git \
    vim \
    nano \
    wget \
    bash-completion
EOF


echo "============ [ installing the python stuff] ============"
ssh -T $USER@$REMOTE_HOST <<EOF
sudo yum install -y python3-pip
cd ~/
pip3 install -U pip
pip3 install -U virtualenv
virtualenv -p python3 ink2023
source ~/ink2023/bin/activate
pip3 install -U \
    pandas \
    numpy \
    sklearn \
    torchvision \
    jupyterlab \
    case-converter \
    ujson \
    fastparquet \
    pyarrow \
    flask \
    tornado \
    huggingface \
    scipy \
    ipykernel \
    ipython
EOF
