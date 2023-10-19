#!/bin/bash

# Install pip if not available
if ! command -v pip &> /dev/null
then
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
fi

# Install Jupyter
pip install jupyter
