#!/bin/bash
paths=$(ls ./data/preprocessed_data | sort)
files=()
for p in $paths; do
    files+=("$(basename "$p")")
done
read -p "test set ratio: " test_ratio
read -p "window size: " window

cd ./src
for ((i=0; i<=500; i++)); do
    echo "第${i}筆   進行 VARMA/ARIMA 處理: ${files[$i]}..."
    python ./VARMA_prediction.py --filename=${files[$i]} --testratio=${test_ratio} --window_size=${window}
done

python ./residual.py --window=${window}
echo "VARMA/ARIMA 階段執行完畢"