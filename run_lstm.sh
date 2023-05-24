cd ./src
python ./lstm_train.py  --double_layer=0 --neurons=32 --dropout=0.5
python ./lstm_pred.py   --double_layer=0 --neurons=32 --dropout=0.5