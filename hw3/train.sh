#!/bin/bash
python3 ./data_process.py $1
KERAS_BACKEND=theano THEANO_FLAGS='floatX=float32,device=gpu,lib.cnmem=0.6,mode=FAST_RUN' python3 semi-auto.py $2 $3