#!/bin/bash
pip install --no-cache -r install/requirements.txt
sudo apt install -y gnupg
sudo apt update
sudo apt install -y gcc g++ cpp
pip install --no-cache cupy-cuda113

cd ./models/archs/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
#python setup.py install --user
python setup_over_cudnn102.py install --user
cd ../../../
