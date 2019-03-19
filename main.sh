#!/usr/bin/env bash

echo "Downloading SimpleQuestions dataset...\n"
wget https://www.dropbox.com/s/9lxudhdfpfkihr1/data.zip
unzip data.zip
rm data.zip

echo "Preprocess the raw data"
python3.6 trim_names.py -f data/freebase-FB2M.txt -n data/FB5M.name.txt

echo "Create processed, augmented dataset...\n"
python3.6 augment_process_dataset.py -d data/


echo "Embed the Knowledge Graph:\n"
echo "It takes too long time and an existing method is used. Thus, we download the Knowledge Graph Embedding directly...\n"
wget https://www.dropbox.com/s/o5hd8lnr5c0l6hj/KGembed.zip
unzip KGembed.zip
rm KGembed.zip
mv -f KGembed/* preprocess/
rm -r KGembed
#python3.6 transE_emb.py --learning_rate 0.003 --batch_size 3000 --eval_freq 50



echo "We could runn train_detection.py, train_entity.py, train_pred.py simultaneously"

echo "Head Entity Detection (HED) model, train and test the model..."
python3.6 train_detection.py --entity_detection_mode LSTM --fix_embed --gpu 0

echo "Entity representation learning..."
python3.6 train_entity.py --qa_mode GRU --fix_embed --gpu 0
python3.6 train_pred.py --qa_mode GRU --fix_embed --gpu 0

echo "We have to run train_detection.py, train_entity.py, train_pred.py first, before running test_main.py..."
python3.6 test_main.py --gpu 0
