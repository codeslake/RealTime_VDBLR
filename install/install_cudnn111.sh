#!/bin/bash
pip install --no-cache -r install/requirements.txt
apt install -y gnupg
apt update
apt install -y gcc g++ cpp
pip install --no-cache cupy-cuda111

cd ./models/archs/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
#python setup.py install --user
python setup_over_cudnn102.py install --user
cd ../../../
