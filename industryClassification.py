import os
import shutil

folder_a = './data/VARMA_ARIMA/after_ARIMA_full'
folder_b = './data/VARMA_ARIMA/after_ARIMA_tsmc'

# check_list = ['2881', '2891', '2886', '2882', '2884', '2885', '2892', '5880', '2887', '2880', '2890', '2883', '2801']  # 金融業 最後共 13 筆，78 時間序列
# check_list = ['2303', '2308', '2317', '2327', '2330', '2357', '2379', '2382', '2395', '2408', '2454', '3008', '3034', '3037', '3711', '6415', '4904', '3045', '2412']  # 含通訊的科技業 最後共 15 筆，105 時間序列
# check_list = ['2303', '2308', '2317', '2327', '2330', '2357', '2379', '2382', '2395', '2408', '2454', '3008', '3034', '3037', '3711', '6415']  # 科技業 最後共 12 筆，66 時間序列
check_list = ['2303', '2330', '2379', '2454']  # 台積電上下游：聯電、聯發科、瑞昱
# check_list = ['2603', '2609', '2615'] # 航運相關
# check_list = ['3045', '2412', '4904'] # 通訊相關
# check_list = ['2002', '1301', '1326', '1101', '1303'] # 石化工業相關，10 時間序列

if not os.path.exists(folder_b):
    os.makedirs(folder_b)


for filename in os.listdir(folder_a):
    if filename.endswith('.csv'):
        parts = filename.split('_')
        print(parts)
        new_list = [item.split('.')[0] if '.' in item else item for item in parts]
        if all(part in check_list for part in new_list):
            source_path = os.path.join(folder_a, filename)
            destination_path = os.path.join(folder_b, filename)
            shutil.copy(source_path, destination_path)
            print("A")
