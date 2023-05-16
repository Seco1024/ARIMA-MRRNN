cd ./src
python ./lstm.py --model=ARIMA --neurons=32 --double_layer=0 --l2=1
python ./lstm.py --model=VARMA --neurons=32 --double_layer=0 --l2=1
# python ./lstm.py --model=ARIMA --neurons=40 --double_layer=0 --l2=1
# python ./lstm.py --model=VARMA --neurons=40 --double_layer=0 --l2=1
# python ./lstm.py --model=ARIMA --neurons=16 --double_layer=0 --l2=1
# python ./lstm.py --model=VARMA --neurons=16 --double_layer=0 --l2=1