#!/bin/bash

# Create data directory
mkdir -p data

# Install fanoutqa and its dependencies
pip install "fanoutqa[eval]"
python -m spacy download en_core_web_sm
pip install "bleurt @ git+https://github.com/google-research/bleurt.git@master"
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
rm BLEURT-20.zip
