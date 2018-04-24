#/bin/bash

source venv/bin/activate
. init-vars.sh

python scoring/1-preprocess.py
python scoring/2.1-create_ensemble1.py
python scoring/2.2-score_ensemble1.py
python scoring/4-predict_ensemble1_ensemble2.py
