import yfinance as yf
from source.auxiliary_functions import remove_outliers_DDC, apply_estimator

# Time series extraction from yahoo finance
# importing the time series

import pandas
import bs4 as bs
import datetime as dt
import os
import numpy
import pandas_datareader.data as web
import pickle
import requests
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import pandas as pd
import seaborn as sns
from hyperimpute.plugins.imputers import Imputers

from source.auxiliary_functions import estimate_cov, contaminate_bernoulli
from sklearn.datasets import load_breast_cancer, load_wine

# Imports as csv files the open, close, high, low and volume of the SP500 stocks
# For our purposes, we will only keep the open and close
# Code from stack overflow (https://stackoverflow.com/questions/58890570/python-yahoo-finance-download-all-sp-500-stocks)
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker[:-1])
    with open("./datasets/sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

def save_nasdaq_tickers():
    data = pd.read_csv('./datasets/nasdaq.csv', sep=',')
    tickers = data['Symbol']
    return tickers

def get_data_from_yahoo(reload_sp500=False, start=dt.datetime(2020,1,1), end=dt.datetime.now(), verbose=0, nasdaq=False):
    if nasdaq:
        tickers = tickers = save_nasdaq_tickers()
        folder = 'stocks_df_nasdaq'    
    elif reload_sp500 or not os.path.exists('./datasets/sp500tickers.pickle'):
        tickers = save_sp500_tickers()
        folder = 'stocks_df_sp500'
    else:
        with open("datasets/sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
        folder = 'stocks_df_sp500'       
        
    if not os.path.exists(folder):
        os.makedirs(folder)

    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists(folder+'/{}.csv'.format(ticker)):
            if ticker == "BRK.B": #strange stock not found on yahoo
                pass
            else:
                try:
                    print('{} : '.format(ticker), end='')
                    df = yf.download(ticker, start=start, end=end)
                    df.reset_index(inplace=True)
                    df.set_index("Date", inplace=True)
                    df.to_csv(folder+'/{}.csv'.format(ticker))
                except:
                    print('Error while loading {}'.format(ticker))
        else:
            if verbose > 0:
                print('Already have {}'.format(ticker))

def load_stocks(columns="Close", nasdaq=False):
    if nasdaq:
        folder = 'stocks_df_nasdaq'
    else:
        folder = 'stocks_df_sp500'
    df = pandas.DataFrame()
    for file in tqdm(os.listdir(folder)):
        stock = pandas.read_csv(folder+"/"+file)
        stock.set_index('Date', inplace=True)
        df = df.join(stock[columns].astype(float), how="outer", rsuffix="_"+file.replace(".csv", ""))
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    return df

def get_sample_matrix():
    df = load_stocks(nasdaq=False)
    df = df.diff()/df
    df.drop(df.index[0], inplace=True)
    df.dropna(how='any', axis=1, inplace=True)
    to_drop = df.eq(0).rolling(10).sum().eq(10).any() # line removing any asset with 10 consecutive 0
    df = df.loc[:, ~to_drop]
    res = df.to_numpy()
    res = res/numpy.std(res, axis=0)
    return res, []

def get_sample_nasdaq():
    df = load_stocks(nasdaq=True)
    to_drop = df.eq(0).rolling(10).sum().eq(10).any() # line removing any asset with 10 consecutive 0
    df = df.loc[:, ~to_drop]
    df = df.diff()/df
    df.drop(df.index[0], inplace=True)
    df.dropna(how='any', axis=1, inplace=True)
    to_drop = df.eq(0).rolling(10).sum().eq(10).any() # line removing any asset with 10 consecutive 0
    df = df.loc[:, ~to_drop]
    res = df.to_numpy()
    res = res/numpy.std(res, axis=0)
    #print(res)
    return res, []

def get_contamination_SP500(start=dt.datetime(2020,1,1), end=dt.datetime(2020,2,1), intensity=0, M=100):
    get_data_from_yahoo(reload_sp500=True, start=start, end=end)
    sample_matrix = get_sample_matrix()
    true_covariance = numpy.cov(sample_matrix.T)
    true_norm = numpy.linalg.norm(true_covariance, ord=2)
    e_rank = numpy.trace(true_covariance)/true_norm
    print("Effective rank of the covariance on this period: {}".format(e_rank))
    _, true_eigs = numpy.linalg.eig(true_covariance)
    
    deltas = numpy.linspace(0, 0.3, 50)

    errors = numpy.zeros((deltas.shape[0], M))

    for i, delta in enumerate(tqdm(deltas)):
        for j in range(M):
            new_mat, _ = contaminate_bernoulli(sample_matrix, delta, intensity=intensity)
            cont_cov = numpy.cov(new_mat.T)
            errors[i, j] = numpy.linalg.norm(cont_cov - true_covariance, ord=2)/true_norm

    return deltas, errors.mean(axis=1), errors.std(axis=1)

#def accuracy_detection_SP500(start=dt.datetime(2020,1,1), end=dt.datetime(2020,2,1), intensity=0, M=100):
#    get_data_from_yahoo(reload_sp500=True, start=start, end=end)
#    sample_matrix = get_sample_matrix()
#
#    for i, delta in enumerate(tqdm(deltas)):
#        for j in range(M):


##### Cleaning datasets #####

def mask_union(A,B):
    if len(A) == 0:
        return B
    if len(B) == 0:
        return A
    if not isinstance(A, numpy.ndarray):
        A = numpy.array(A)
    if not isinstance(B, numpy.ndarray):
        B = numpy.array(B)
    return A + B - A*B

def detect_nan(X):
    return numpy.isnan(X).astype(int)

def clean_abalone():
    data = pd.read_csv('./datasets/abalone.data', sep=',', header=None)
    data.drop([0,8], axis=1, inplace=True)
    # dropping sample wise outliers (can do due to sample size)
    data = data[ data[3] < 0.30 ]
    data = data[ data[3] > 0.001 ]
    data = data[ data[1] > 0.001 ]
    data = data[ data[2] > 0.001 ]
    data = data.apply(lambda x: numpy.sqrt(x))
    data = data.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()))
    return data.to_numpy(), []

def clean_cameras():
    data = pd.read_csv('./datasets/cameras.csv', sep=';')
    types = data.iloc[0]
    data.drop(0, axis=0, inplace=True)
    data.drop(['Model', 'Release date'], axis=1, inplace=True)
    data=data.astype(float)

    data['Zoom tele (T)'] = numpy.log(data['Zoom tele (T)']+1)
    data['Macro focus range'] = numpy.log(data['Macro focus range']+1)
    data['Storage included'] = numpy.log(data['Storage included']+1)
    data['Weight (inc. batteries)'] = numpy.log(data['Weight (inc. batteries)']+1)
    data['Price'] = numpy.log(data['Price'])

    data_numpy = data.to_numpy()
    nan_mask = detect_nan(data_numpy)
    data.fillna(0, inplace=True)
    data = data.to_numpy()

    data = (data - numpy.nanmean(data, axis=0))/numpy.nanstd(data, axis=0)

    return data, []

def clean_breast_cancer():
    data, _ = load_breast_cancer(return_X_y=True)
    data = (data - numpy.mean(data, axis=0))/numpy.std(data, axis=0)
    return data, []

def clean_wine():
    data, _ = load_wine(return_X_y=True)
    data = (data - numpy.mean(data, axis=0))/numpy.std(data, axis=0)
    return data, []

# woolridge datasets:
def clean_barium():
    data = pd.read_csv('./datasets/BARIUM.raw', delimiter=r"\s+", header=None)
    data.replace({'.': 0}, inplace=True)
    categorical_cols = [2,3,4,5,6,7,11,12,13,19,20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    data.drop(categorical_cols, axis=1, inplace=True)
    log_cols = [0, 1, 10, 16, 30]
    for c in log_cols:
        data[c] = numpy.log(data[c])
    data = (data - numpy.nanmean(data, axis=0))/numpy.nanstd(data, axis=0)
    return data.to_numpy(), []

def clean_attend():
    data = pd.read_csv('./datasets/ATTEND.raw', delimiter=r"\s+", header=None)
    data.replace({'.': 0}, inplace=True)
    data.replace({'.': 0}, inplace=True)
    categorical_cols = [6,7,8]
    data.drop(categorical_cols, axis=1, inplace=True)
    log_cols = [ 9]
    for c in log_cols:
        data[c] = numpy.log(data[c]+1)
    exp_cols = [0, 5]
    for c in exp_cols:
        data[c] = numpy.exp(numpy.sqrt(data[c]))
    data = (data - numpy.nanmean(data, axis=0))/numpy.nanstd(data, axis=0)
    return data, []

def clean_ceosal():
    data = pd.read_csv('./datasets/CEOSAL2.raw', delimiter=r"\s+", header=None)
    data.replace({'.': 0}, inplace=True)
    data.replace({'.': 0}, inplace=True)
    categorical_cols = [2,3]
    data.drop(categorical_cols, axis=1, inplace=True)
    log_cols = [0, 5, 6,7,8, 11, 13]
    for c in log_cols:
        data[c] = numpy.log(data[c]+1)
    sqrt_cols=[12]
    for c in sqrt_cols:
        data[c] = numpy.sqrt(data[c])
    data = (data - numpy.nanmean(data, axis=0))/numpy.nanstd(data, axis=0)
    return data, []

def clean_hseinv():
    data = pd.read_csv('./datasets/HSEINV.raw', delimiter=r"\s+", header=None)
    data.replace({'.': 0}, inplace=True)
    data = (data - numpy.nanmean(data, axis=0))/numpy.nanstd(data, axis=0)
    return data

def clean_htv():
    data = pd.read_csv('./datasets/HTV.raw', delimiter=r"\s+", header=None)
    data.replace({'.': 0}, inplace=True)
    data.replace({'.': 0}, inplace=True)
    categorical_cols = [3,4,5,6,10, 12, 13, 14, 15, 16, 17, 19, 20, 7] # the last three are redundant
    data.drop(categorical_cols, axis=1, inplace=True)
    log_cols = [0]
    for c in log_cols:
        data[c] = numpy.log(data[c]+1)
    sqrt_cols=[21]
    for c in sqrt_cols:
        data[c] = numpy.sqrt(data[c])
    data = (data - numpy.nanmean(data, axis=0))/numpy.nanstd(data, axis=0)
    return data, []

def clean_intdef():
    data = pd.read_csv('./datasets/INTDEF.raw', delimiter=r"\s+", header=None)
    data.replace({'.': 0}, inplace=True)
    data.replace({'.': 0}, inplace=True)
    categorical_cols = [12]
    data.drop(categorical_cols, axis=1, inplace=True)
    log_cols = []
    for c in log_cols:
        data[c] = numpy.log(data[c]+1)
    sqrt_cols=[]
    for c in sqrt_cols:
        data[c] = numpy.sqrt(data[c])
    data = data.astype(float)
    data = (data - numpy.nanmean(data, axis=0))/numpy.nanstd(data, axis=0)
    return data, []

def clean_kielmc():
    data = pd.read_csv('./datasets/KIELMC.raw', delimiter=r"\s+", header=None)
    data.replace({'.': 0}, inplace=True)
    data.replace({'.': 0}, inplace=True)
    categorical_cols = [0, 10, 11, 16 , 21, 22]
    data.drop(categorical_cols, axis=1, inplace=True)
    log_cols = [1, 2, 7,23]
    for c in log_cols:
        data[c] = numpy.log(data[c]+1)
    sqrt_cols=[]
    for c in sqrt_cols:
        data[c] = numpy.sqrt(data[c])
    # many missing values on dimension 19
    data = (data - numpy.nanmean(data, axis=0))/numpy.nanstd(data, axis=0)
    return data, []

def rel_dist(A,B):
    if A is None or B is None:
        return None
    norm = max(numpy.linalg.norm(A, ord=2), numpy.linalg.norm(B, ord=2))
    return numpy.linalg.norm(A-B, ord=2)/norm

def test_estimators(fun):
    data, nan_mask = fun()
    print("dim : ", data.shape[1])
    print("nsamples : ", data.shape[0])
    data_mv = numpy.nan_to_num(data)
    classical = numpy.cov(data_mv.T)
    print("erank : ", numpy.trace(classical)/ numpy.linalg.norm(classical, ord=2))

    ddc_mask_99, _ = remove_outliers_DDC(data)
    ddc_mask_95, _ = remove_outliers_DDC(data, 0.95)
    mask_99 = mask_union(ddc_mask_99, nan_mask)
    mask_95 = mask_union(ddc_mask_95, nan_mask)

    try:
        cov_ddc_99, _ = apply_estimator('oracleMV', data, 1-mask_99)
    except:
        cov_ddc_99 = None
    try :
        cov_ddc_95, _ = apply_estimator('oracleMV', data, 1-mask_95)
    except:
        cov_ddc_95 = None
    try:
        cov_ii, _ = apply_estimator('DDCII', data, mask_99)
    except:
        cov_ii = None
    try:
        cov_knn, _ = apply_estimator('DDCKNN', data, mask_99)
    except:
        cov_knn = None
    try:
        cov_tsgs, _ = apply_estimator('TSGS', data)
    except:
        cov_tsgs = None
    try:
        cov_di, _ = apply_estimator('DI', data)
    except:
        cov_di = None

    results = pandas.DataFrame('-', columns=['DDC MV 99','DDC MV 95', 'DDC II 99', 'DDC KNN 99', 'TSGS', 'DI'],
                                index=['classical','DDC MV 99','DDC MV 95', 'DDC II 99', 'DDC KNN 99', 'TSGS'])
    results.loc['classical','DDC MV 99'] = rel_dist(classical, cov_ddc_99)
    results.loc['classical','DDC MV 95'] = rel_dist(classical, cov_ddc_95)
    results.loc['classical','DDC II 99'] = rel_dist(classical, cov_ii)
    results.loc['classical','DDC KNN 99'] = rel_dist(classical, cov_knn)
    results.loc['classical','TSGS'] = rel_dist(classical, cov_tsgs)
    results.loc['classical','DI'] = rel_dist(classical, cov_di)
    results.loc['DDC MV 99','DDC MV 95'] = rel_dist(cov_ddc_99, cov_ddc_95)
    results.loc['DDC MV 99','DDC II 99'] = rel_dist(cov_ddc_99, cov_ii)
    results.loc['DDC MV 99','DDC KNN 99'] = rel_dist(cov_ddc_99, cov_knn)
    results.loc['DDC MV 99','TSGS'] = rel_dist(cov_ddc_99, cov_tsgs)
    results.loc['DDC MV 99','DI'] = rel_dist(cov_ddc_99, cov_di)
    results.loc['DDC MV 95','DDC II 99'] = rel_dist(cov_ddc_95, cov_ii)
    results.loc['DDC MV 95','DDC KNN 99'] = rel_dist(cov_ddc_95, cov_knn)
    results.loc['DDC MV 95','TSGS'] = rel_dist(cov_ddc_95, cov_tsgs)
    results.loc['DDC MV 95','DI'] = rel_dist(cov_ddc_95, cov_di)
    results.loc['DDC II 99','DDC KNN 99'] = rel_dist(cov_ii, cov_knn)
    results.loc['DDC II 99','TSGS'] = rel_dist(cov_ii, cov_tsgs)
    results.loc['DDC II 99','DI'] = rel_dist(cov_ii, cov_di)
    results.loc['DDC KNN 99','TSGS'] = rel_dist(cov_knn, cov_tsgs)
    results.loc['DDC KNN 99','DI'] = rel_dist(cov_knn, cov_di)
    results.loc['TSGS','DI'] = rel_dist(cov_tsgs, cov_di)

    return results

def test_estimators_heatmap(fun):
    data, nan_mask = fun()
    data_mv = numpy.nan_to_num(data)
    classical = numpy.cov(data_mv.T)

    ddc_mask_99, _ = remove_outliers_DDC(data)
    ddc_mask_95, _ = remove_outliers_DDC(data, 0.95)
    mask_99 = mask_union(ddc_mask_99, nan_mask)
    mask_95 = mask_union(ddc_mask_95, nan_mask)

    try:
        cov_ddc_99, _ = apply_estimator('oracleMV', data, 1-mask_99)
    except:
        cov_ddc_99 = None
    try :
        cov_ddc_95, _ = apply_estimator('oracleMV', data, 1-mask_95)
    except:
        cov_ddc_95=None
    try:
        cov_ii, _ = apply_estimator('DDCII', data, mask_99)
    except:
        cov_ii = None
    cov_ii = None
    try:
        cov_knn, _ = apply_estimator('DDCKNN', data, mask_99)
    except:
        cov_knn = None
    try:
        cov_tsgs, _ = apply_estimator('TSGS', data)
    except:
        cov_tsgs=None
    try:
        cov_di, _ = apply_estimator('DI', data)
    except:
        cov_di = None


    data_nan = data*ddc_mask_99
    data_nan[data_nan == 0] = numpy.nan

    gain = Imputers().get('gain')
    x_gain = gain.fit_transform(data_nan)
    cov_gain = numpy.cov(x_gain.T)

    miwae = Imputers().get('miwae')
    x_miwae = miwae.fit_transform(data_nan)
    cov_miwae = numpy.cov(x_miwae.T)


    results = pandas.DataFrame(None, columns=['DDC MV 99','DDC MV 95', 'DDC KNN 99', 'DDC II 99', 'DDC MIWAE','DDC GAIN','TSGS', 'DI'],
                                index=['classical','DDC MV 99','DDC MV 95', 'DDC KNN 99', 'DDC II 99','DDC MIWAE','DDC GAIN','TSGS'])
    results.loc['classical','DDC MV 99'] = rel_dist(classical, cov_ddc_99)
    results.loc['classical','DDC MV 95'] = rel_dist(classical, cov_ddc_95)
    results.loc['classical','DDC II 99'] = rel_dist(classical, cov_ii)
    results.loc['classical','DDC KNN 99'] = rel_dist(classical, cov_knn)
    results.loc['classical','DDC MIWAE'] = rel_dist(classical, cov_miwae)
    results.loc['classical','DDC GAIN'] = rel_dist(classical, cov_gain)
    results.loc['classical','TSGS'] = rel_dist(classical, cov_tsgs)
    results.loc['DDC MV 99','DDC MV 95'] = rel_dist(cov_ddc_99, cov_ddc_95)
    results.loc['DDC MV 99','DDC II 99'] = rel_dist(cov_ddc_99, cov_ii)
    results.loc['DDC MV 99','DDC KNN 99'] = rel_dist(cov_ddc_99, cov_knn)
    results.loc['DDC MV 99','DDC MIWAE'] = rel_dist(cov_ddc_99, cov_miwae)
    results.loc['DDC MV 99','DDC GAIN'] = rel_dist(cov_ddc_99, cov_gain)
    results.loc['DDC MV 99','TSGS'] = rel_dist(cov_ddc_99, cov_tsgs)
    results.loc['DDC MV 95','DDC II 99'] = rel_dist(cov_ddc_95, cov_ii)
    results.loc['DDC MV 95','DDC KNN 99'] = rel_dist(cov_ddc_95, cov_knn)
    results.loc['DDC MV 95','DDC MIWAE'] = rel_dist(cov_ddc_95, cov_miwae)
    results.loc['DDC MV 95','DDC GAIN'] = rel_dist(cov_ddc_95, cov_gain)
    results.loc['DDC MV 95','TSGS'] = rel_dist(cov_ddc_95, cov_tsgs)
    results.loc['DDC KNN 99', 'DDC II 99'] = rel_dist(cov_ii, cov_knn)
    results.loc['DDC KNN 99','DDC MIWAE'] = rel_dist(cov_knn, cov_miwae)
    results.loc['DDC KNN 99','DDC GAIN'] = rel_dist(cov_knn, cov_gain)
    results.loc['DDC II 99','DDC MIWAE'] = rel_dist(cov_ii, cov_miwae)
    results.loc['DDC II 99','DDC GAIN'] = rel_dist(cov_ii, cov_gain)
    results.loc['DDC II 99','TSGS'] = rel_dist(cov_ii, cov_tsgs)
    results.loc['DDC MIWAE','DDC GAIN'] = rel_dist(cov_miwae, cov_gain)
    results.loc['DDC MIWAE','TSGS'] = rel_dist(cov_miwae, cov_tsgs)
    results.loc['DDC GAIN','TSGS'] = rel_dist(cov_gain, cov_tsgs)
    results.loc['DDC KNN 99','TSGS'] = rel_dist(cov_knn, cov_tsgs)

    try:
        results.loc['classical','DI'] = rel_dist(classical, cov_di)
        results.loc['DDC MV 99','DI'] = rel_dist(cov_ddc_99, cov_di)
        results.loc['DDC MV 95','DI'] = rel_dist(cov_ddc_95, cov_di)
        results.loc['DDC II 99','DI'] = rel_dist(cov_ii, cov_di)
        results.loc['DDC KNN 99','DI'] = rel_dist(cov_knn, cov_di)
        results.loc['DDC MIWAE','DI'] = rel_dist(cov_miwae, cov_di)
        results.loc['DDC GAIN','DI'] = rel_dist(cov_gain, cov_di)
        results.loc['TSGS','DI'] = rel_dist(cov_tsgs, cov_di)
    except:
        pass

    results = results.astype(float)

    return results