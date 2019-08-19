import numpy as np
import pandas as pd
from datetime import datetime
from fastparquet import write
import matplotlib.pyplot as plt

import finml as fn
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import dash_table



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# initialize the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# load data
global_df = pd.read_parquet('data_dollarbars.pq')

# Create app layout
app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    
    html.Div([
        
        html.Div([
            dcc.Graph(id='pie-chart')
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),    
            
        html.Div([
            html.Label('Holding Period (Hr.)'),
            dcc.Slider(
                id='holding-period-slider',
                min=1,
                max=3,
                value=1,
                marks={str(hr): str(hr) for hr in [1, 1.5, 2, 2.5, 3]},
                step=0.5,
            ),
        
            html.Br(),        
            html.Label('Profit Target Multiple'),        
            dcc.Slider(
                id='profit-target-multiplier',
                min=1,
                max=3,
                value=2,
                marks={str(hr): str(hr) for hr in [1, 1.5, 2, 2.5, 3]},
                step=0.5,
            ),
                
            html.Br(),
            html.Label('Stop-Loss Multiple'),
            dcc.Slider(
                id='stop-loss-multiplier',
                min=1,
                max=3,
                value=2,
                marks={str(hr): str(hr) for hr in [1, 1.5, 2, 2.5, 3]},
                step=0.5,
            ),        
        ], style={'width': '15%', 'display': 'inline-block',
                  'vertical-align': 'top'}),

    
    ]),
    
    
        
    
    # Hidden div inside teh app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})
    
])
        

@app.callback(
        Output('intermediate-value', 'children'),
        [Input('holding-period-slider', 'value'),
         Input('profit-target-multiplier', 'value'),
         Input('stop-loss-multiplier', 'value')]
        )    
def update_samples(holding_period, pt, sl):
    
    # to be updated
    h = pd.Timedelta(hours=holding_period)  
    m = [pt, sl]
    
    # add holding period return
    updated_df = global_df.assign(retp1 = fn.get_return(global_df.close)).dropna()
    
    # add thresholds and vertical barrier (t1) columns
    updated_df = updated_df.assign(threshold=fn.get_volatility(updated_df.close, delta=h), 
                             t1=fn.get_verticals(updated_df, delta=h)).dropna()
    
    # events are [t1, threshold, side]
    events = updated_df[['t1', 'threshold']]
    events = events.assign(side=pd.Series(1., events.index)) # long only

    # get the timestamps for [t1, stop_loss, take_profit]
    touches = fn.get_horizontals(updated_df.close, events, m)
    # assign labels based on which barrier is hit first
    touches = fn.get_labels(touches)
    
    # add touches timestamps and label
    updated_df = pd.concat( [updated_df.loc[:, 'volume':'threshold'], 
                        touches.loc[:, 't1':'label']], axis=1)
    
    
    # more generally, this line would be
    # json.dumps(updated_df)
    return updated_df.to_json(date_format='iso', date_unit='ns', orient='split')


@app.callback(
        Output('graph-with-slider', 'figure'),
        [Input('intermediate-value', 'children')]
        )
def update_graph(jsonified_updated_data):
    
    # more generally, this line would be
    # json.loads(jsonified_updated_data)
    dff = pd.read_json(jsonified_updated_data, orient='split')
    
    # only plot last 8 hours
    end = dff.index[-1]
    start = end - pd.Timedelta(hours=8)
    
    filtered_dff = dff[start:end]
    
    # label indices
    pos_idx = filtered_dff[filtered_dff.label == 1].index
    neg_idx = filtered_dff[filtered_dff.label == -1].index
    zero_idx = filtered_dff[filtered_dff.label == 0].index  
    
    figure={
            'data': [
                { 
                    'x': filtered_dff.index, 
                    'y': filtered_dff.close, 
                    'type': 'line',
                    'name': 'price',

                },
                   
                { 
                    'x': pos_idx, 
                    'y': filtered_dff.close.loc[pos_idx], 
                    'name': '+1',
                    'mode': 'markers',
                    'marker': {'size': 5, 'color': 'green'}
                },
                
                { 
                    'x': neg_idx, 
                    'y': filtered_dff.close.loc[neg_idx], 
                    'name': '-1',
                    'mode': 'markers',
                    'marker': {'size': 5, 'color': 'red'}
                },
                
                { 
                    'x': zero_idx, 
                    'y': filtered_dff.close.loc[zero_idx], 
                    'name': '0',
                    'mode': 'markers',
                    'marker': {'size': 5, 'color': 'blue'}
                }
            ],
            #'layout': {'margin': {'b': 50, 'r': 10, 'l': 30, 't': 10}},
                
         }

    return figure


@app.callback(
        Output('pie-chart', 'figure'),
        [Input('intermediate-value', 'children')]
        )
def update_pie(jsonified_updated_data):
    
    dff = pd.read_json(jsonified_updated_data, orient='split')
    cnt = dff.label.value_counts()
    labels = list(cnt.index)
    values = list(cnt.values)

    pie=go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.3)],
            )
    pie.update_layout(legend_orientation='h')
    
    return pie
    


##======= Modeling =============

## columns to keep and features
#cols = ['volume','open', 'high', 'low', 'close', 'retp1', 'threshold']
#
## split data into training and test sets
#X = data_ohlc[cols].values
#y = np.squeeze(data_ohlc[['label']].values)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
#X.shape, X_train.shape, X_test.shape
#
## classifier
#rf = RandomForestClassifier()

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
#rf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', 
#                            criterion='gini', random_state=42)
#
## fit and predict
#rf.fit(X_train, y_train)
#y_pred = rf.predict(X_test)
#
## prediction accuracy
#rf.score(X_test, y_test)
#
#pd.Series(y_pred).value_counts()
     
         
if __name__ == '__main__':
    app.run_server(debug=True)