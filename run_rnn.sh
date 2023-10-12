cd ./src
# python ./self_Attention_train.py --dropout=0.5 --head_size=32 --lookback=30 --epochs=40
# python ./self_Attention_pred.py --dropout=0.5 --head_size=32 --lookback=30 
python ./mrrnn_train.py  --neurons=256 --dropout=0.5 --cell="lstm" --lookback=60 --epochs=200
python ./rnn_train.py    --neurons=256 --dropout=0.5 --cell="LSTM" --lookback=60 --epochs=200
python ./mrrnn_train.py  --neurons=256 --dropout=0.5 --cell="gru" --lookback=60 --epochs=200
python ./rnn_train.py    --neurons=256 --dropout=0.5 --cell="GRU" --lookback=60 --epochs=200
# python ./mrrnn_train.py  --neurons=256 --dropout=0.5 --cell="rnn" --lookback=15 --epochs=200