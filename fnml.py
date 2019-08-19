import numpy as np
import pandas as pd
from datetime import datetime
import ciso8601
import glob
import pickle
from fastparquet import write
import time


def _compute_vwap(df):
    """
    Compute volume weighted average price.
    
    Parameters:
    ----------
    df: pd.DataFrame
    """
    q = df['foreignNotional']
    p = df['price']
    vwap = np.sum(p * q) / np.sum(q)
    df['vwap'] = vwap
    return df


def _ohlc(df):
    """
    Compute OHLC
    
    Parameters:
    -----------
    df: pd.DataFrame
    
    Output:
    -------
    pd.DataFrame
    """
    df['open'] = df.price.iloc[0]
    df['high'] = df.price.max()
    df['low'] = df.price.min()
    df['close'] = df.price.iloc[-1]
    df['volume'] = df['size'].sum()   
    df['trades'] = df['size'].count()
    return df[-1:]


def _get_ts_lag(time_series, lag=pd.Timedelta(hours=1)):
    """Compute time series at given lag"""
    # find time_series[t-1] integer indices given lag
    df0 = time_series.index.searchsorted(time_series.index - lag)
    df0 = df0[df0 > 0]  
    
    # align time_series[t-1] timestamps to time_series[t] timestamps 
    df0 = pd.Series(time_series.index[df0 - 1],
                   index=time_series.index[time_series.shape[0] - df0.shape[0] : ])
    
    df0 = pd.Series(time_series[df0.values].values, index=df0.index)
    
    return df0


def _get_direction(prices):
    d1 = prices.diff(1)
    d2 = prices.diff(2)
    td = np.where(np.isnan(d1), np.nan,
             np.where(d1 > 0, "PlusTick",
                      np.where(d1 < 0, "MinusTick",
                               np.where(np.isnan(d2), np.nan,
                                        np.where(d2 > 0, "zeroPlusTick", "zeroMinusTick")))))
    return td

def dollar_sampling(df, dollars_per_bar = 2e6):
    """
    Sampling observations based on a pre-defined exchanged market value
    
    Parameters
    ----------
    df: pd.DataFrame 
    dollars_per_bar: pre-defined dollar amount to sample
    
    Output:
    df: pd.DataFrame
    """
    
    # add cumulative dollar column
    data_cm_dollar = df.assign(cmDollar=df['foreignNotional'].cumsum())
    
    # compute total_dollars
    total_dollars = data_cm_dollar.cmDollar.values[-1]
    
    # group trade by cmDollar//dollars_per_bar as groupId
    data_dollar_grp = data_cm_dollar.assign(grpId=lambda row: row.cmDollar // dollars_per_bar)
    
    # for each groupId, compute vwap, OHLC, volume, and number of trades
    data_dollar_ohlc =  data_dollar_grp.groupby('grpId').apply(lambda x: _ohlc(_compute_vwap(x)))
    
    # drop index level
    data_dollar_ohlc.index = data_dollar_ohlc.index.droplevel()
    
    # drop rows with duplicated index but keep the first occurence
    data_dollar_ohlc = data_dollar_ohlc[~data_dollar_ohlc.index.duplicated(keep='first')]
    
    # keep columns
    mask = ['vwap', 'open', 'high', 'low', 'close', 'volume', 'trades']
    data_dollar_ohlc = data_dollar_ohlc[mask]
    
    return data_dollar_ohlc


def _get_volatility(prices, span=100, delta=pd.Timedelta(hours=1)):
    """
    Compute price return of the form p[t]/p[t-1] - 1
    
    Input: prices :: pd series of prices
           span0  :: the width or lag of the ewm() filter
           delta  :: time interval of volatility to be computed
    Output: pd series of volatility for each given time interval
    """
    
    # find p[t-1] indices given delta
    df0 = prices.index.searchsorted(prices.index - delta)
    df0 = df0[df0 > 0]  
    
    # align p[t-1] timestamps to p[t] timestamps 
    df0 = pd.Series(prices.index[df0 - 1],
                   index=prices.index[prices.shape[0] - df0.shape[0] : ])
    
    # get values for each timestamps then compute returns
    df0 = prices.loc[df0.index] / prices.loc[df0.values].values - 1
    
    # estimate rolling standard deviation
    df0 = df0.ewm(span=span).std()
    df0 = df0[df0 != 0]
    
    return df0


##==== Functions implementing Triple-Barrier Method ====
    
def _get_verticals(prices, delta=pd.Timedelta(hours=1)):
    """
    Returns the timestamps for vertical barriers given
    a strategy's holding period.
    
    Input:  prices :: pd series of prices
            delta  :: strategy's holding period
    Output: pd Series of timestamps for vertical barriers
    
    Implement code snippet 3.4 in "Advances in Financial Machines Learning"
    by Marcos Lopez De Padro.    
    """
    
    # find the vertical barrier index for each timestamp
    t1 = prices.index.searchsorted(prices.index + delta)
    t1 = t1[t1 < prices.shape[0]] 
    
    # retrieve the vertical barrier's timestamp
    t1 = prices.index[t1]
    
    # as a series
    t1 = pd.Series(t1, index=prices.index[:t1.shape[0]])
    return t1

def _get_horizontals(data, factors=[2, 2]):
    
    # events are [t1, threshold, side]
    events = data[['t1', 'threshold']]
    events = events.assign(side=pd.Series(1., events.index)) # long only
    
    out = events[['t1']].copy(deep=True)
    
    # set upper threshold
    if factors[0] > 0:
        thresh_upper = factors[0] * events['threshold']
    else: 
        thresh_upper = pd.Series(index=events.index)         # NaN; no upper threshold
        
    # set lower threshold
    if factors[1] > 0:
        thresh_lower = -factors[1] * events['threshold']
    else:
        thresh_lower = pd.Series(index=events.index)         # NaN; no lower threshold
    
    # return the timestamp of earliest stop-loss or profit taking
    for loc, t1 in events['t1'].iteritems():  
        dfhi = data['high'][loc:t1]                                  # path of high prices
        dflo = data['low'][loc:t1]                                   # path of low prices
        dfhi = (dfhi / data['close'][loc] - 1) * events.side[loc]    # path of high returns
        dflo = (dflo / data['close'][loc] - 1) * events.side[loc]    # path of low returns
        out.loc[loc, 'stop_loss'] = dflo[dflo < thresh_lower[loc]].index.min()   # earliest stop loss
        out.loc[loc, 'take_profit'] = dfhi[dfhi > thresh_upper[loc]].index.min() # earliest profit taking
    return out

def _get_labels(touches):
    """
    Assigns a label in {-1, 0, 1} depending on which of the 
    three barriers is hit first. 
    
    Input: touches:: get_horizontals(data_ohlc.close, events, [1,1]),
                     a dataframe with three columns:
                     t1            :: the timestamp of vertical barrier
                     stop_loss     :: the timestamp for the lower barrier
                     profit_taking :: the time stamp for the upper barrier
    The result of get_horizon() will then be used to assign labels based on
    which barrier is hit first.
    
    Based on Maks Ivanov's implementation of MLDP's triple-barrier labeling method.
    """
    
    out = touches.copy(deep=True)
    # pandas df.min() ignores NaN values
    first_touch = touches[['stop_loss', 'take_profit']].min(axis=1)
    for loc, t in first_touch.items():
        if pd.isnull(t):
            out.loc[loc, 'label'] = 0
        elif t == touches.loc[loc, 'stop_loss']:
            out.loc[loc, 'label'] = -1
        else:
            out.loc[loc, 'label'] = 1
    return out


def assign_labels(data, holding_period=pd.Timedelta(hours=1),\
                  volatility_window=pd.Timedelta(hours=1), factors = [2, 2]):
    """
    Implement Triple-Barrier labeling method in AFML by Marcos Lopez De Padro 
    but with modified method to compute upper and lower horizontal bounds.
    """

    data = data.assign(tickDirection=_get_direction(data.close),
                                   closeLag1Hr=_get_ts_lag(data.close, lag=pd.Timedelta(hours=1)))
    data = data.assign(return1Hr=data.closeLag1Hr/data.close - 1).dropna()

    # add thresholds and vertical barrier (t1) columns
    data = data.assign(threshold=_get_volatility(data.close, delta=volatility_window), 
                             t1=_get_verticals(data, delta=holding_period)).dropna()

    # events are [t1, threshold, side]
    events = data[['t1', 'threshold']]
    events = events.assign(side=pd.Series(1., events.index)) # long only

    # get the timestamps for [t1, stop_loss, take_profit]
    touches = _get_horizontals(data, factors)
    # assign labels based on which barrier is hit first
    touches = _get_labels(touches)

    # add touches timestamps and label
    data = pd.concat( [data.loc[:, 'vwap':'threshold'], 
                        touches.loc[:, 't1':'label']], axis=1)
    
    return data