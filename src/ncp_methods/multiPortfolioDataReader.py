from pyomo.environ import * 
from pyomo.mpec import *
import pyomo.kernel as pmo
import numpy as np 
import pandas as pd
import os,time
import gurobipy as gp
from gurobipy import GRB	
from numpy import linalg
import scipy as sc
import openpyxl,random
from typing import Any, Dict, Self

class multiPortfolioDataReader:
    """
    Class to read in stock return data from a CSV file and generate random 
    multi-portfolio optimization instances.

    Attributes:
        datafile (str): Path to the CSV file containing stock return data.
    """

    def __init__(self: Self, datafile: str) -> None:
        """
        Initializes the multiPortfolioDataReader with a given data file.
        Args:
            datafile (str): Path to the CSV file containing stock return data.
        """
        self.datafile = datafile

    def getStockList(self: Self) -> list[str]:
        """
        Retrieves the list of stock tickers from the data file.

        Returns:
            list[str]: A list of stock tickers.
        """
        df = pd.read_csv(self.datafile,skiprows=1,index_col=0)
        for s in df.columns:
            df[s] = df[s].str.rstrip('%').astype('float') / 100.0
        df = df.transpose()
        stocks = df.index.tolist()
        return stocks

    def getStockData(self: Self, stocks: list[str]) -> pd.DataFrame:
        """
        Retrieves the stock return data for the specified list of stocks.
        Args:
            stocks (list[str]): List of stock tickers to retrieve data for.
        Returns:
            pd.DataFrame: A DataFrame containing the stock return data 
                for the specified tickers.
        """
        df = pd.read_csv(self.datafile,skiprows=1,index_col=0).dropna().drop('10-year average')[stocks]
        for s in stocks:
            df[s] = df[s].str.rstrip('%').astype('float') /100.0
        df = df.transpose().sort_index(inplace=False)
        return df
    
    def generateInstance(
            self: Self, 
            numPortfolios: int, 
            assetsPerPortfolio: int, 
            numFirms: int
        ) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
        """
        Generates random multi-portfolio optimization instances.
        Args:
            numPortfolios (int): Number of portfolios to generate.
            assetsPerPortfolio (int): Number of assets in each portfolio.
            numFirms (int): Number of firms to generate.
        Returns:
            tuple: A tuple containing two dictionaries:
                - portfolio (dict): A dictionary where each key is a portfolio index
                    and the value is another dictionary with keys 'assets', 'returns',
                    'mu', and 'V'.
                - firm (dict): A dictionary where each key is a firm index and the value
                    is another dictionary with keys 'B', 'eps', 'R', 'mat', and 'F_inv'.
        """
        portfolio = {}
        
        for p in range(numPortfolios):
            assets = random.sample(self.getStockList(),k=assetsPerPortfolio)
            returns = self.getStockData(assets).to_numpy()
            mu = np.mean(returns,axis=1)
            

            cov = np.cov(returns)
            if p == 0:
                allReturns = returns
            else: 
                allReturns = np.concatenate((allReturns,returns),axis=0)
            

            portfolio[p] = {
                'assets': assets,
                'returns': returns,
                'mu': mu,
                'V': np.cov(allReturns)
            }
        params = sc.stats.fit(sc.stats.norm,allReturns.flatten())
        f_inv = sc.stats.norm.ppf(0.30,loc=np.mean(allReturns),scale=np.std(allReturns)) 
        firm = {}
        #test = np.mean(allReturns,axis=1)
        for f in range(numFirms):
            budget = assetsPerPortfolio*numPortfolios
            eps = 0.1
            r_min = np.min(np.mean(allReturns,axis=1))
            f_inv = f_inv
            mat = {}
            for p in range(numPortfolios):
                B = np.random.rand(assetsPerPortfolio,assetsPerPortfolio)
                mat[p] = B@B.T/assetsPerPortfolio

            firm[f] = {
                'B': budget,
                'eps': eps,
                'R':r_min,
                'mat': mat,
                'F_inv': f_inv
            } 

            
        return portfolio,firm