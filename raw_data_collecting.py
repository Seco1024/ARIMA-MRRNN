import requests
import pandas as pd
from datetime import date, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv, find_dotenv

retry_strategy = Retry(total=3)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)
load_dotenv(find_dotenv())

# get etf50
response = http.get("https://www.moneydj.com/ETF/X/Basic/Basic0007a.xdjhtm?etfid=0050.TW")
response.encoding = 'utf-8'

## parse html
soup = BeautifulSoup(response.text, "html.parser")
etf50_df = pd.DataFrame()
row_index = 0

## locate the table by find the sibling html tag which have id attribute
first_table = soup.find(id="ctl00_ctl00_MainContent_MainContent_sdate3").find_next_sibling()
stock_tag = first_table.find_all("td")
for i in range(0, len(stock_tag), 4):
    stock_name = stock_tag[i].text.strip()
    etf50_df.loc[row_index, "stock_name"] = stock_name
    etf50_df.loc[row_index, "持股(千股)"] = stock_tag[i + 1].text.strip()
    etf50_df.loc[row_index, "比例"] = stock_tag[i + 2].text.strip()
    etf50_df.loc[row_index, "增減"] = stock_tag[i + 3].text.strip()
    row_index += 1

stock_tag = first_table.find_next_sibling().find_all("td")
for i in range(0, len(stock_tag), 4):
    stock_name = stock_tag[i].text.strip()
    etf50_df.loc[row_index, "stock_name"] = stock_name
    etf50_df.loc[row_index, "持股(千股)"] = stock_tag[i + 1].text.strip()
    etf50_df.loc[row_index, "比例"] = stock_tag[i + 2].text.strip()
    etf50_df.loc[row_index, "增減"] = stock_tag[i + 3].text.strip()
    row_index += 1

# login to finmind
url = "https://api.finmindtrade.com/api/v4/data"
login_url = "https://api.finmindtrade.com/api/v4/login"
login_payload = {
    "user_id": os.getenv("USER_ID"),
    "password": os.getenv("PASSWORD")
}
login_data = requests.post(login_url, data=login_payload)
token = login_data.json()["token"]

# get etf50 tickers data
ticker_params = {
    "dataset": "TaiwanStockInfo",
    "token":token,
}
ticker_data = http.get(url, params=ticker_params).json()
ticker_df = pd.DataFrame(ticker_data["data"])
result_df = pd.merge(ticker_df, etf50_df, how="inner", on=["stock_name"])
result_df = result_df.drop_duplicates(subset=['stock_name'])
result_df = result_df.reset_index(drop=True)
result_df.to_csv("./data/raw_data/etf50_tickers.csv", index=False, header=True, encoding='big5')

# get technical data
etf50_df = pd.read_csv("./data/raw_data/etf50_tickers.csv", encoding="big5")
individual_technical_dataset = []

for id in etf50_df["stock_id"]:
    technical_params = {
        "dataset": "TaiwanStockPrice",
        "data_id": id,
        "start_date": "2013-03-01",
        "end_date": "2023-03-01",
        "token": token, 
    }
    technical_data = requests.get(url, params=technical_params).json()
    tmp_df = pd.DataFrame(technical_data["data"])
    tmp_df['date'] = pd.to_datetime(tmp_df['date'], format='%Y-%m-%d')
    tmp_df.set_index('date', inplace=True)
    tmp_df.to_csv(f'./data/raw_data/Technical/{id}_technical.csv')
    individual_technical_dataset.append(tmp_df)

technical_df = pd.concat(individual_technical_dataset)
technical_df = technical_df.reset_index(drop=True)
technical_df.to_csv('./data/raw_data/Technical/etf50_technical.csv')

# get margin and short sell data
individual_mtss_dataset = []
for id in etf50_df["stock_id"]:
    mtss_params = {
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "data_id": id,
        "start_date": "2013-03-01",
        "end_date": "2023-03-01",
        "token": token, 
    }
    mtss_data = requests.get(url, params=mtss_params).json()
    tmp_df = pd.DataFrame(mtss_data["data"])
    tmp_df = tmp_df.loc[:, ["date", "stock_id", "MarginPurchaseBuy", "MarginPurchaseSell", "ShortSaleBuy", "ShortSaleSell"]]
    tmp_df['date'] = pd.to_datetime(tmp_df['date'], format='%Y-%m-%d')
    tmp_df.set_index('date', inplace=True)
    tmp_df.to_csv(f'./data/raw_data/Margin_Short_Sell/{id}_mtss.csv')
    individual_mtss_dataset.append(tmp_df)

mtss_df = pd.concat(individual_mtss_dataset)
mtss_df = mtss_df.reset_index(drop=True)
mtss_df.to_csv('./data/raw_data/Margin_Short_Sell/etf50_mtss.csv')

# get major investor institution data
individual_mii_dataset = []

for id in etf50_df["stock_id"]:
    mii_params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": id,
        "start_date": "2013-03-01",
        "end_date": "2023-03-01",
        "token": token, 
    }

    mii_data = requests.get(url, params=mii_params).json()
    tmp_df = pd.DataFrame(mii_data["data"])
    pivot_df = tmp_df.pivot_table(index=['date','stock_id'], columns='name', values=['sell', 'buy'])
    pivot_df.columns = pivot_df.columns.map(lambda x: f'{x[1]}_{x[0]}' if isinstance(x, tuple) else x)
    pivot_df = pivot_df.fillna(0)
    columns = pivot_df.columns.get_level_values(0)

    if 'Dealer_buy' not in columns:
        pivot_df['Dealer_buy'] = pivot_df['Dealer_self_buy'] + pivot_df['Dealer_Hedging_buy'] 
        pivot_df['Dealer_sell'] = pivot_df['Dealer_self_sell'] + pivot_df['Dealer_Hedging_sell']    
    else:
        pivot_df['Dealer_buy'] = pivot_df['Dealer_buy'] + pivot_df['Dealer_Hedging_buy'] + pivot_df['Dealer_self_buy']    
        pivot_df['Dealer_sell'] = pivot_df['Dealer_sell'] + pivot_df['Dealer_Hedging_sell'] + pivot_df['Dealer_self_sell'] 

    pivot_df['Foreign_Investor_buy'] = pivot_df['Foreign_Dealer_Self_buy'] + pivot_df['Foreign_Investor_buy'] 
    pivot_df['Foreign_Investor_sell'] = pivot_df['Foreign_Dealer_Self_sell'] + pivot_df['Foreign_Investor_sell']
    pivot_df = pivot_df.drop(['Dealer_Hedging_buy', 'Dealer_self_buy', 'Foreign_Dealer_Self_buy', 'Dealer_Hedging_sell', 'Dealer_self_sell', 'Foreign_Dealer_Self_sell'], axis = 1)  
    pivot_df[['Dealer_buy', 'Dealer_sell', 'Foreign_Investor_buy', 'Foreign_Investor_sell', 'Investment_Trust_buy', 'Investment_Trust_sell']] = pivot_df[['Dealer_buy', 'Dealer_sell', 'Foreign_Investor_buy', 'Foreign_Investor_sell', 'Investment_Trust_buy', 'Investment_Trust_sell']].astype('int64')
    pivot_df = pivot_df.rename({'_date': 'date', '_stock_id': 'stock_id'}, axis=1)
    pivot_df = pivot_df.reset_index()
    pivot_df['date'] = pd.to_datetime(pivot_df['date'], format='%Y-%m-%d')
    pivot_df.set_index('date', inplace=True)
    pivot_df.to_csv(f'./data/raw_data/Major_Investor/{id}_mii.csv')
    individual_mii_dataset.append(pivot_df)

mii_df = pd.concat(individual_mii_dataset)     
mii_df = mii_df.reset_index(drop=True)
mii_df.to_csv('./data/raw_data/Major_Investor/etf50_mii.csv')

print("Complete Data Collecting")