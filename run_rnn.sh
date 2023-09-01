cd ./src
python ./rnn_train.py  --double_layer=0 --neurons=256 --dropout=0.5 --cell="GRU"
python ./rnn_pred.py   --double_layer=0 --neurons=256 --dropout=0.5 --cell="GRU"
# python ./self_Attention_train.py --dropout=0.5
# python ./self_Attention_pred.py --dropout=0.5
python ./mrrnn_train.py  --neurons=256 --dropout=0.5 --cell="rnn"
python ./mrrnn_pred.py   --neurons=256 --dropout=0.5 --cell="rnn"