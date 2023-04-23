#!/bin/bash
chmod u+x run_VARMA.sh
TRUE=1
read -p "是否進行資料前處理, 請輸入0或1:" preprocess
if [ ${preprocess} == ${TRUE} ];then
    read -p "請輸入 dropna 閥值 (預設=0.01): " drop_ratio
    read -p "請輸入 stride 大小 (預設=100): " stride
    read -p "請輸入動態窗口大小 (預設=100): " window
    echo "執行資料前處理中......"
    python ./data_preprocessing.py --drop=${drop_ratio} --stride=${stride} --window=${window}
    echo "完成資料前處理"
fi
dataset_n=$(ls ./data/preprocessed_data | wc -l)
read -p "請輸入欲執行 VARMA/ARIMA 處理的次數比例: " ratio
n=$(echo "${dataset_n} * ${ratio}" | bc)
paths=$(ls ./data/preprocessed_data | shuf | head -n ${n%.*})
files=()
for p in $paths; do
    files+=("$(basename "$p")")
done
read -p "請輸入測試集比例: " test_ratio
for f in ${files[@]}; do
    echo "進行 VARMA/ARIMA處理: ${f}..."
    python ./VARMA_prediction.py --filename=${f} --testratio=${test_ratio}
done
python ./residual.py
echo "VARMA/ARIMA 執行完畢"