cd ./src

python ./lstm_train.py --neurons=64 --double_layer=0 --l2=1
python ./lstm_pred.py  --neurons=64 --double_layer=0 --l2=1