#!/usr/bin/env python
"""
This file allows to communicate with the R packages implementing robust statistics

Authors: Karim Lounici and Grégoire Pacreau
"""

import subprocess
import pandas as pd
import numpy as np
import os

def DDC(data, quantile=0.99, return_residuals=False):
    # Data is a pandas Dataframe
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_csv("temp/DDC_data.csv", index=False)

    # Calls the script which produces a boolean matrix detecting outliers using DDC
    os.system("Rscript R_scripts/DDC.R temp/DDC_data.csv {} {}".format(quantile, return_residuals))

    res = pd.read_csv("temp/DDC_data_res.csv").to_numpy()
    os.remove("temp/DDC_data.csv")
    os.remove("temp/DDC_data_res.csv")
    return res

def TSGS(data, filter="UBF-DDC", partial_impute=False, tol=1e-4, maxiter=150, method="bisquare", init="emve"):
    # Data is a pandas Dataframe
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_csv("temp/TSGS_data.csv", index=False)

    # Calls the script which produces a robust covariance matrix
    os.system("Rscript R_scripts/TSGS.R temp/TSGS_data.csv {} {} {} {} {} {}".format(
        str(filter), str(partial_impute), str(tol), str(maxiter), method, init
    ))

    res_mu = pd.read_csv("temp/TSGS_data_res_mu.csv").to_numpy()
    res_S = pd.read_csv("temp/TSGS_data_res_S.csv").to_numpy()
    res_data = pd.read_csv("temp/TSGS_data_res_filtered.csv").to_numpy()
    os.remove("temp/TSGS_data.csv")
    os.remove("temp/TSGS_data_res_mu.csv")
    os.remove("temp/TSGS_data_res_S.csv")
    os.remove("temp/TSGS_data_res_filtered.csv")
    return res_mu, res_S, res_data

def DI(data, initEst="DDCWcov", crit=0.01, maxits=10, quant=0.99, maxCol=0.25):
    # Data is a pandas Dataframe
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_csv("temp/DI_data.csv", index=False)

    # Calls the script which produces a robust covariance matrix
    os.system("Rscript R_scripts/DI.R temp/DI_data.csv {} {} {} {} {}".format(
        str(initEst), str(crit), str(maxits), str(quant), str(maxCol)
    ))

    res_mu = pd.read_csv("temp/DI_data_res_mu.csv").to_numpy()
    res_S = pd.read_csv("temp/DI_data_res_S.csv").to_numpy()
    os.remove("temp/DI_data.csv")
    os.remove("temp/DI_data_res_mu.csv")
    os.remove("temp/DI_data_res_S.csv")
    return res_mu, res_S

def DDCwcov(data, maxCol=0.25):
    # NOT WORKING
    # Data is a pandas Dataframe
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.to_csv("temp/DDCwcov_data.csv", index=False)

    # Calls the script which produces a robust covariance matrix
    os.system("Rscript R_scripts/DDCwcov.R temp/DDCwcov_data.csv " + str(maxCol))

    res = pd.read_csv("temp/DDCwcov_data_res.csv")
    os.remove("temp/DDCwcov_data.csv")
    os.remove("temp/DDCwcov_data_res.csv")
    return res

if __name__ == "__main__":
    data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=50)

    print("DDC")
    res = DDC(data)
    print(res)

    print("TSGS")
    _, res, _ = TSGS(data)
    print(res)

    print("DI")
    _, res = DI(data)
    print(res)

