#!/bin/bash
TRUE=1
cd ./src
read -p "是否進行資料前處理, 請輸入0或1:" preprocess
if [ ${preprocess} == ${TRUE} ];then
    read -p "dropna threshold (default=0.01): " drop_ratio
    read -p "window size (default=100): " window
    read -p "stride size (default=100): " stride
    read -p "features (0 for default, 1 for major, 2 for technical): " mode
    echo "執行資料前處理中......"
    python ./data_preprocessing.py --drop=${drop_ratio} --stride=${stride} --window=${window} --mode=${mode}
    echo "完成資料前處理"
fi

cd ../
dataset_n=$(ls ./data/preprocessed_data | wc -l)
read -p "欲執行 VARMA/ARIMA 處理的次數比例: " ratio
n=$(echo "${dataset_n} * ${ratio}" | bc)
paths=$(ls ./data/preprocessed_data | shuf | head -n ${n%.*})
files=()
for p in $paths; do
    files+=("$(basename "$p")")
done

cd ./src
read -p "test set ratio: " test_ratio
read -p "window size: " window
for f in ${files[@]}; do
    echo "進行 ARIMA 處理: ${f}..."
    python ./arima_prediction.py --filename=${f} --testratio=${test_ratio} --window_size=${window}
done
python ./residual.py --window=${window}
echo "ARIMA 執行完畢"