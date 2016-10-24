#!/bin/bash
python ./logistic_regression.py ./spam_data/spam_train.csv model
python ./test.py model ./spam_data/spam_test.csv result.csv
