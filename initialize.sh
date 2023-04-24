mkdir ./data
cd ./data
mkdir ./raw_data
mkdir ./preprocessed_data
mkdir ./VARMA_ARIMA
mkdir ./plot
cd ./raw_data
mkdir Major_Investor
mkdir Margin_Short_Sell
mkdir Technical
cd ../VARMA_ARIMA
mkdir ./after_ARIMA
mkdir ./after_VARMA
cd ../plot
mkdir ./after_ARIMA
mkdir ./after_VARMA
cd ../../

mkdir ./out
cd ./out
mkdir ./VARMA_ARIMA_residual
cd ./VARMA_ARIMA_residual
mkdir ./anomalies
cd ../../

conda install requirements.txt
python raw_data_collecting.py