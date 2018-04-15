#!/bin/bash

virtualenv venv
source venv/bin/activate

pip install pip --upgrade
pip install setuptools --upgrade
pip install -r dependencies.txt
