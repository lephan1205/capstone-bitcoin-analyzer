import numpy as np
import pandas as pd
from datetime import datetime
from fastparquet import write
import matplotlib.pyplot as plt

import finml as fn
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

##====== Load the data and process ========

## Specify holding period return and horizontal multipliers
h = pd.Timedelta(hours=1)
m =[2, 2]


# load the data with OHLC
data_ohlc = pd.read_parquet('data_dollarbars.pq')

# add holding period return
data_ohlc = data_ohlc.assign(retp1 = fn.get_return(data_ohlc.close)).dropna()

# add thresholds and vertical barrier (t1) columns
data_ohlc = data_ohlc.assign(threshold=fn.get_volatility(data_ohlc.close, delta=h), 
                             t1=fn.get_verticals(data_ohlc, delta=h)).dropna()

# events are [t1, threshold, side]
events = data_ohlc[['t1', 'threshold']]
events = events.assign(side=pd.Series(1., events.index)) # long only

# get the timestamps for [t1, stop_loss, take_profit]
touches = fn.get_horizontals(data_ohlc.close, events, m)
# assign labels based on which barrier is hit first
touches = fn.get_labels(touches)

# add label column to dataframe
# data_ohlc = data_ohlc.assign(label=touches.label)

# add touches timestamps and label
data_ohlc = pd.concat( [data_ohlc.loc[:, 'volume':'threshold'], 
                        touches.loc[:, 't1':'label']], axis=1)


data_ohlc.shape

# look at the distribution of labels
data_ohlc.label.value_counts()

# trades by day
tradelog = pd.DataFrame(index=data_ohlc.index)
tradelog = tradelog.assign(day=tradelog.index.weekday_name, time=tradelog.index.time)

##======= Modeling =============

# columns to keep and features
cols = ['volume','open', 'high', 'low', 'close', 'retp1', 'threshold']

# split data into training and test sets
X = data_ohlc[cols].values
y = np.squeeze(data_ohlc[['label']].values)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
X.shape, X_train.shape, X_test.shape

# classifier
rf = RandomForestClassifier()

## Grid search for tuning parameters
#rf_grid = GridSearchCV(
#    rf,
#    {"n_estimators": [100, 125, 150],
#     "max_depth" : [8, 10, 12],
#     "criterion" : ['gini', 'entropy']},
#    cv=5,     # 5-fold cross-validation
#    n_jobs=2, # run each hyperparameter in one of two parallel jobs
#)
#rf_grid.fit(X_train, y_train)
#print(rf_grid.best_params_)

# Predict with random forest using balanced resampling
rf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', 
                            criterion='gini', random_state=42)

# fit and predict
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# prediction accuracy
rf.score(X_test, y_test)

pd.Series(y_pred).value_counts()


##====== Dash Visualization Workflow ===========

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction
import plotly.graph_objs as go
import dash_table


# plot window and data
start = pd.Timestamp(year=2019, month=7, day=10, hour=9)
end = pd.Timestamp(year=2019, month=7, day=10, hour=13)

data = data_ohlc.loc[start:end]
pos_idx = data[data.label == 1].index
neg_idx = data[data.label == -1].index
zero_idx = data[data.label == 0].index

# transition probability matrix
M = fn.get_trans_probability(data_ohlc.label)
M_n = pd.DataFrame(np.linalg.matrix_power(M, 10), index=M.index, columns=M.columns).round(decimals=2)



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# initialize the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Create global chart template
layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    ),


# Create app layout
app.layout = html.Div(children=
    [
         html.H3(
            children="BitCoin Trade Analyzer",
            style={"margin-bottom": "0px"}
         ),
         
         html.Div([ 
             dcc.Graph(
                 id="price-chart",
                 figure={
                    'data': [
                        { 
                            'x': data.index, 
                            'y': data.close, 
                            'type': 'line',
                            'name': 'price',

                        },
                           
                        { 
                            'x': pos_idx, 
                            'y': data.close.loc[pos_idx], 
                            'name': '+1',
                            'mode': 'markers',
                            'marker': {'size': 5, 'color': 'green'}
                        },
                        
                        { 
                            'x': neg_idx, 
                            'y': data.close.loc[neg_idx], 
                            'name': '-1',
                            'mode': 'markers',
                            'marker': {'size': 5, 'color': 'red'}
                        },
                        
                        { 
                            'x': zero_idx, 
                            'y': data.close.loc[zero_idx], 
                            'name': '0',
                            'mode': 'markers',
                            'marker': {'size': 5, 'color': 'blue'}
                        }
                        
                    ]
                 }
            )
         ]),

        html.Div([
            html.Div([
                dash_table.DataTable(
                    id='transition-matrix',
                    columns=[ {"name": str(i), "id": str(i)} for i in M_n.columns ],
                    data=M_n.to_dict('records'),
                ),
                
                
            ], style={'width': '30%', 'display': 'inline-block'}),
                
            html.Div([
                dcc.Graph(
                    id='trade-scatter',
                    figure={
                         'data': [
                            {
                                'x': tradelog.time,
                                'y': tradelog.day
                            },
                        ]   
                    }
                )        
            ], style={'width': '70%', 'display': 'inline-block'})
                
        ])

     ]
)
         
         
if __name__ == '__main__':
    app.run_server(debug=True)