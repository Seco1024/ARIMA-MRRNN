cd ./src
# python ./lstm_train.py  --double_layer=0 --neurons=32 --dropout=0.5
# python ./lstm_pred.py   --double_layer=0 --neurons=32 --dropout=0.5
# python ./self_Attention_train.py --dropout=0.5
# python ./self_Attention_pred.py --dropout=0.5
python ./mrrnn_train.py  --neurons=32 --dropout=0.5
# python ./mrrnn_pred.py   --neurons=32 --dropout=0.5