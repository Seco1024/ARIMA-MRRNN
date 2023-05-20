cd ./src

python ./lstm_train.py --neurons=32 --double_layer=0 --l2=0.002 --dropout=0.2
python ./lstm_pred.py  --neurons=32 --double_layer=0 --l2=0.002 --dropout=0.2