#!/usr/bin/env bash

echo "Downloading SimpleQuestions dataset...\n"
wget https://www.dropbox.com/s/9lxudhdfpfkihr1/data.zip
unzip data.zip

echo "Downloading Knowledge Graph Embedding...\n"
wget https://www.dropbox.com/s/90l0xony07s1ybq/preprocess.zip
unzip preprocess.zip

echo "Preprocess the raw data"
python3.6 trim_names.py -f data/freebase-FB2M.txt -n data/FB5M.name.txt

echo "\n\nCreate processed, augmented dataset...\n"
python3.6 augment_process_dataset.py -d data/


echo "Entity Detection, Train and test the model"
python3.6 train_detection.py --entity_detection_mode LSTM --fix_embed --gpu 0

echo "Entity representation learning"
python3.6 train_entity.py --qa_mode LSTM --fix_embed --gpu 0
python3.6 train_pred.py --qa_mode LSTM --fix_embed --gpu 0
python3.6 test_main.py --gpu 0
