import numpy as np
import pandas as pd
from datetime import datetime
from fastparquet import write
import matplotlib.pyplot as plt
import networkx as nx


##===== Functions to process data ====

def compute_vwap(df):
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


def ohlc(df):
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
    data_dollar_ohlc =  data_dollar_grp.groupby('grpId').apply(lambda x: ohlc(compute_vwap(x)))
    
    # drop index level
    data_dollar_ohlc.index = data_dollar_ohlc.index.droplevel()
    
    # drop rows with duplicated index but keep the first occurence
    data_dollar_ohlc = data_dollar_ohlc[~data_dollar_ohlc.index.duplicated(keep='first')]
    
    # keep columns
    mask = ['vwap', 'open', 'high', 'low', 'close', 'volume', 'trades']
    data_dollar_ohlc = data_dollar_ohlc[mask]
    
    return data_dollar_ohlc


def get_return(prices, span=100, delta=pd.Timedelta(hours=1)):
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
    ret = prices.loc[df0.index] / prices.loc[df0.values].values - 1
    
    return ret

def get_volatility(prices, span=100, delta=pd.Timedelta(hours=1)):
    """
    Compute volatility.
    """
    ret = get_return(prices, span, delta)
    vol = ret.ewm(span=span).std()
    return vol


def get_trans_probability(sequence):
    
    G = nx.DiGraph()

    for (i, j) in zip(sequence, sequence[1:]):
        if (i,j) in G.edges():
            G[i][j]['weight'] += 1
        else:
            G.add_edge(i, j, weight=1)

    for n in G.nodes:
        node_total = 0
        for (u, v, wt) in G.edges.data('weight'):
            if u == n:
                node_total += wt
        for (u, v, wt) in G.edges.data('weight'):
            if u == n:
                G[u][v]['pct'] = wt/node_total
                
    df = pd.DataFrame(index=G.nodes, columns=G.nodes)
    for edge in G.edges:
        df.loc[edge[0], edge[1]] = G[edge[0]][edge[1]]['pct']
    df.fillna(value=0, inplace=True)
    
    return df.round(decimals=2)

##==== Functions implementing Triple-Barrier Method ====
    
def get_verticals(prices, delta=pd.Timedelta(hours=1)):
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


def get_horizontals(prices, events, factors=[2, 2]):
    """
    Apply profit taking/stop loss based on volatility estimate, 
    if it takes place before t1 (vertical barrier). Return the
    timestamp of earliest stop-loss or profit taking.
    
    Input: prices  ::  pd series of prices
           events  ::  pd dataframe with 3 columns:
                       t1          ::: the timestamp of vertical barriers; 
                                       if t1=np.nan then no vertical barriers
                       thresholds  ::: unit height of top and bottom barriers
                       side        ::: the side of each bet; side = 1 is long, side = -1 is short
           factors ::  multipliers ::: threshold multiplier to set the height of top/bottom barriers
        
    Output: pd dataframe with 3 columns:
            t1            :: the timestamp of vertical barrier
            stop_loss     :: the timestamp for the lower barrier
            profit_taking :: the time stamp for the upper barrier
    
    The result of get_horizon() will then be used in get_labels() to assign labels based on
    which barrier is hit first.  
    
    NOTE: to get "events", we first assign ""thresholds" and "t1" columns to data_ohlc dataframe
    using get_volatility() for thresholds and get_verticals() for t1. That is:
        
           data_ohlc = data_ohlc.assign(threshold=get_vol(data_ohlc.close)).dropna()
           data_ohlc = data_ohlc.assign(t1=get_horizons(data_ohlc)).dropna()
           events = data_ohlc[['t1', 'threshold']] 
           events = events.assign(side=pd.Series(1., events.index))
    
    Implement code snippet 3.2 in "Advances in Financial Machines Learning"
    by Marcos Lopez De Padro.  
    
    A good read to understand AFML implementation is by Maks Ivanov on
    https://towardsdatascience.com/financial-machine-learning-part-1-labels-7eeed050f32e.   
    """ 
    
    out = events[['t1']].copy(deep=True)
    
    # set upper threshold
    if factors[0] > 0:
        thresh_upper = factors[0] * events['threshold']
    else: 
        thresh_upper = pd.Series(index=events.index)      # NaN; no upper threshold
        
    # set lower threshold
    if factors[1] > 0:
        thresh_lower = -factors[1] * events['threshold']
    else:
        thresh_lower = pd.Series(index=events.index)      # NaN; no lower threshold
    
    # return the timestamp of earliest stop-loss or profit taking
    for loc, t1 in events['t1'].iteritems():              
        df0 = prices[loc:t1]                              # path prices
        df0 = (df0 / prices[loc] - 1) * events.side[loc]  # path returns
        out.loc[loc, 'stop_loss'] = df0[df0 < thresh_lower[loc]].index.min()   # earliest stop loss
        out.loc[loc, 'take_profit'] = df0[df0 > thresh_upper[loc]].index.min() # earliest profit taking
    return out


def get_labels(touches):
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


def labeling(df, h = pd.Timedelta(hours=1), m = [2, 2]):
    """
    Implement Triple-Barrier labeling method in 
    "Advances in Financial Machines Learning" by Marcos Lopez De Padro.    
    
    Parameters
    ----------
    df:  pd.DataFrame
    h:   Timedelta for holding period; default 1 hour
    m:   list of integers [pt, Sl] for profit taking and stop-loss limit
         as multiple of dynamic volatility    
    """

    # add thresholds and vertical barrier (t1) columns
    df = df.assign(threshold=get_volatility(df.close, delta=h), 
                             t1=get_verticals(df, delta=h)).dropna()
    
    # events are [t1, threshold, side]
    events = df[['t1', 'threshold']]
    events = events.assign(side=pd.Series(1., events.index)) # long only
    
    # get the timestamps for [t1, stop_loss, take_profit]
    touches = get_horizontals(df.close, events, m)
    # assign labels based on which barrier is hit first
    touches = get_labels(touches)
    
    # add touches timestamps and label
    df = pd.concat( [df.loc[:, 'vwap':'threshold'], 
                        touches.loc[:, 't1':'label']], axis=1)
    
    return df 
