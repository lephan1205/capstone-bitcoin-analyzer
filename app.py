
import numpy as np
import pandas as pd
import datetime as dt

import pickle
from scipy.stats import rayleigh
import plotly.graph_objs as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_table



##------Unpickle data and model--------

labeled_df = pd.read_parquet('labeled_data.pq')

# Select columns to build model
dropcols = ['closeLag1Hr', 'threshold', 't1', 'stop_loss', 'take_profit']
X = labeled_df.drop(dropcols, axis=1)
y = labeled_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# load the model from disk
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# dash table data
table_cols = ['open', 'high', 'low', 'close', 'label']
table = X[table_cols].reset_index()
y_pred = loaded_model.predict(X)
table['prediction'] = y_pred




app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

app.layout = html.Div(
    [
         # header
         html.Div(
            [
                html.Div(
                    [
                        html.H6("BITCOIN TRADE ANALYZER", className="app_header_title"),
#                        html.P(
#                                "This app analyzes historical trades and predict price direction.",
#                                className="app__header__title--grey",
#                        ),
                    ],
                    className="app__header__desc",
                ),
            ],
            className="app__header",
            style={'marginBottom': 2, 'marginTop': 5}
        ),   
        html.Div(
            [
                # price chart
                html.Div(
                    [
                        html.Div(
                            [
#                                html.Div(
#                                    [html.H6(id="item1_text"), html.P("Item No. 1")],
#                                    id="item1",
#                                    className="mini_container",
#                                ),
#                                html.Div(
#                                    [html.H6(id="item2_text"), html.P("Item No. 2")],
#                                    id="item2",
#                                    className="mini_container",
#                                ),
#                                html.Div(
#                                    [html.H6(id="item3_text"), html.P("Item No. 3")],
#                                    id="item3",
#                                    className="mini_container",
#                                ),
#                                html.Div(
#                                    [html.H6(id="item4_text"), html.P("Item No. 4")],
#                                    id="item4",
#                                    className="mini_container",
#                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        dcc.Graph(
                            id="price-chart",
                            figure=go.Figure(
                                layout=go.Layout(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                            config={'displayModeBar': False},
                        ),
#                        dcc.Interval(
#                            id="price-chart-update",
#                            interval=int(GRAPH_INTERVAL),
#                            n_intervals=1000000,
#                        ),        
                        dcc.RangeSlider(
                            id="day-slider",
                            min=0,
                            max=len(labeled_df.index)-1,
                            step=1,
                            value=[len(labeled_df.index)-300 , len(labeled_df.index)],
                            marks={ int(i): str(labeled_df.index[int(i)].date()) 
                            for i in list(np.arange(0, 1.2, .2) * (len(labeled_df.index)-1)) }
                        ),
                        html.Div(id='output-container-range-slider-non-linear', style={'margin-top': 40}),
                        
                        html.Div(
                            [
                                dash_table.DataTable(
                                    id='table',
                                    columns=[{"name": i, "id": i} for i in table.columns],
                                    data=table.to_dict("rows"),
                                    style_table={
                                        'height': '150px',
                                        'overflowY': 'scroll',
                                        'border': 'thin lightgrey solid'
                                    },
                                    style_header={
                                        'backgroundColor': '#D7DBDD',
                                        'fontWeight': 'bold',
                                        'color': '#515A5A',
                                    },
                                    style_cell={
                                        # all three widths are needed
                                        'width': '200px', 'width': '80px', 'width': '80px', 'width': '80px',
                                        'whiteSpace': 'no-wrap',
                                        'overflow': 'hidden',
                                        #'textOverflow': 'ellipsis',
                                        'background-color': app_color["graph_bg"],
                                        'color': '#707B7C',                                        
                                    },
                                )
                            ]
                        )
                        
                    ],
                    className="two-thirds column price__chart__container",
                    
                ),
                
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "PRICE HISTOGRAM",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                html.Div(
                                    [
                                        dcc.Slider(
                                            id="bin-slider",
                                            min=1,
                                            max=60,
                                            step=1,
                                            value=20,
                                            updatemode="drag",
                                            marks={
                                                20: {"label": "20"},
                                                40: {"label": "40"},
                                                60: {"label": "60"},
                                            },
                                        )
                                    ],
                                    className="slider",
                                ),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="bin-auto",
                                            options=[
                                                {"label": "Auto", "value": "Auto"}
                                            ],
                                            value=["Auto"],
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                        ),
                                        html.P(
                                            "# of Bins: Auto",
                                            id="bin-size",
                                            className="auto__p",
                                        )
                                    ],
                                    className="auto__container",
                                ),
                                dcc.Graph(
                                    id="price-histogram",
                                    figure=go.Figure(
                                        layout=go.Layout(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                    config={'displayModeBar': False}
                                    
                                ),
                            ],
                            className="graph__container first"
                        ),
                        # pie
                        html.Div(
                            [
                                 html.Div(
                                    [
                                        html.H6(
                                            "ACTUAL SHARE", className="graph__title"
                                        )        
                                    ]
                                ),
                                dcc.Graph(
                                    id="pie",
                                    figure=go.Figure(
                                        layout=go.Layout(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                    config={'displayModeBar': False},
                                ),
                            ],
                            className="graph__container second",
                        ), 
                    ],
                    className="one-third column histogram_pie",
                    
                ),
                    
            ],
            className="app__content",
        ),        
                    
     ],
     className="app__container",
     style={'marginLeft': 10, 'marginRight': 10}
     
)

@app.callback(
    Output("price-chart", "figure"),
    [Input("day-slider", "value")]
)               
def update_chart(selected_day):
    """Display price chart with labels."""
    
    dff = labeled_df[selected_day[0]:selected_day[1]]
    
    # labeling indices
    pos_idx = dff[dff.label == 1].index
    neg_idx = dff[dff.label == -1].index
    zero_idx = dff[dff.label == 0].index 
    
    # create traces
    trace1 = go.Scatter(
        y=dff.close, 
        x=dff.index,
        mode="lines",
        line={"color": "#42C4F7", "width": 1},
        name="price",
        hoverinfo="skip",
    )
    trace2 = go.Scatter(
        y=dff.close.loc[pos_idx],
        x=pos_idx,
        mode="markers",
        marker={"size": 4, "color": "#33ff00"},
        name="+1"
    )
    trace3 = go.Scatter(
        y=dff.close.loc[neg_idx],
        x=neg_idx,
        mode="markers",
        marker={"size": 4, "color": "#ff0000"},
        name="-1"
    )
    trace4 = go.Scatter(
        y=dff.close.loc[zero_idx],
        x=zero_idx,
        mode="markers",
        marker={"size": 4, "color": "#0033ff"},
        name="0"
    )
    
    
    data = [trace1, trace2, trace3, trace4]
    
    return {"data": data,
            "layout": go.Layout(
                        plot_bgcolor=app_color["graph_bg"],
                        paper_bgcolor=app_color["graph_bg"],
                        font={"color": "#fff"},
                        height=400,
                        margin={'t': 5, 'b': 50},
                        autosize=True,
                        xaxis={
                            "showline": True,
                            "zeroline": False,
                            "fixedrange": True,
                            #"tickvals": [0, 50, 100, 150, 200],
                            #"ticktext": ["200", "150", "100", "50", "0"],
                            },
                        yaxis={
                            "showgrid": True,
                            "showline": True,
                            "fixedrange": True,
                            "zeroline": False,
                            "gridcolor": app_color["graph_line"],
                            }
                        ),
            }
           
@app.callback(
    [Output("table", "data"), Output('table', 'columns')],
    [Input("day-slider", "value")]
)    
def update_table(selected_day):
    "Display updated table"
    table_update = table[selected_day[0]:selected_day[1]].to_dict('records')
    columns=[{"name": i, "id": i} for i in table.columns]    
    return table_update, columns
#            
            
#@app.callback(
#    Output("price-histogram", "figure"),
##    [Input("price-chart-update", "n_intervals")],
#    [
#         State("price-chart", "figure"),
#         State("bin-slider", "value"),
#         State("bin-auto", "value"),
#     ],
#)
#def update_histogram(interval, price_chart, slider_value, auto_state):
#    """Display price histogram"""
#    
    
@app.callback(
    Output("price-histogram", "figure"),
    [Input("price-chart", "figure"),
     Input("bin-slider", "value"),
     Input("bin-auto", "value")]
)
def update_histogram(price_chart, slider_value, auto_state):

    price = []
    
    try:
        # Check to see whether price-chart has been plotted
        if price_chart is not None:
            price = price_chart["data"][0]["y"]
        if "Auto" in auto_state:
            bin_val = np.histogram(
                price,                                                       
                bins=range(int(round(min(price))), int(round(max(price)))),
            )
        else:
            bin_val = np.histogram(price, bins=slider_value)
    except Exception as error:
        raise PreventUpdate
        
    avg_val = float(sum(price)) / len(price)
    median_val = np.median(price)
    
    pdf_fitted = rayleigh.pdf(
        bin_val[1], loc=(avg_val) * 0.55, scale=(bin_val[1][-1] - bin_val[1][0]) / 3
    )

    
    y_val = (pdf_fitted * max(bin_val[0]) * 20, )
    y_val_max = max(y_val[0])
    bin_val_max = max(bin_val[0])

    trace = go.Bar(
        x=bin_val[1],
        y=bin_val[0],
        marker={"color": app_color["graph_line"]},
        showlegend=False,
        hoverinfo="x+y",
    )
    
    traces_scatter = [
        {"line_dash": "dash", "line_color": "#2E5266", "name": "Average"},
        {"line_dash": "dot", "line_color": "#BD9391", "name": "Median"},
    ]
                
    scatter_data = [
        go.Scatter(
            x=[bin_val[int(len(bin_val) / 2)]],
            y=[0],
            mode="lines",
            line={"dash": traces["line_dash"], "color": traces["line_color"]},
            marker={"opacity": 0},
            visible=True,
            name=traces["name"],
        )
        for traces in traces_scatter
    ]

#    trace3 = go.Scatter(
#        mode="lines",
#        line={"color": "#42C4F7"},
#        y=y_val[0],
#        x=bin_val[1][: len(bin_val[1])],
#        name="Rayleigh Fit",
#    )
    
    data=[trace, scatter_data[0], scatter_data[1]]    
    
    return {"data": data,
            "layout": go.Layout(
                    height=300,
                    plot_bgcolor=app_color["graph_bg"],
                    paper_bgcolor=app_color["graph_bg"],
                    font={"color": "#fff"},
                    margin={'t': 5, 'b': 20, 'l': 60, 'r': 60},
                    autosize=True,
                    xaxis={
                        #"title": "Price",
                        "showgrid": False,
                        "showline": False,
                        "fixedrange": True,
                    },
                    yaxis={
                        "showgrid": False,
                        "showline": False,
                        "zeroline": False,
                        "title": "Number of Samples",
                        "fixedrange": True,
                    },
                    bargap=0.01,
                    bargroupgap=0,
                    hovermode="closest",
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "xanchor": "center",
                        "y": 1,
                        "x": 0.5,
                    },
                    shapes=[
                        {
                            "xref": "x",
                            "yref": "y",
                            "y1": int(max(bin_val_max, y_val_max)) + 0.5,
                            "y0": 0,
                            "x0": avg_val,
                            "x1": avg_val,
                            "type": "line",
                            "line": {"dash": "dash", "color": "#2E5266", "width": 5},
                        },
                        {
                            "xref": "x",
                            "yref": "y",
                            "y1": int(max(bin_val_max, y_val_max)) + 0.5,
                            "y0": 0,
                            "x0": median_val,
                            "x1": median_val,
                            "type": "line",
                            "line": {"dash": "dot", "color": "#BD9391", "width": 5},
                        },
                    ],
                )
                
            }
    
@app.callback(
    Output("bin-auto", "value"),
    [Input("bin-slider", "value")],
    [State("price-chart", "figure")],
)
def deselect_auto(slider_value, price_chart):
    """Toggle the auto checkbox."""
    
    # prevent if graph has no data
    if not len(price_chart["data"]):
        raise PreventUpdate
        
    if price_chart is not None:
        return [""]
    
    return ["Auto"]


@app.callback(
    Output("bin-size", "children"),
    [Input("bin-auto", "value")],
    [State("bin-slider", "value")]        
)
def show_num_bins(autoValue, slider_value):
    """Display the number of bins."""
    
    if "Auto" in autoValue:
        return "# of Bins: Auto"
    return "# of Bins: " + str(int(slider_value))


@app.callback(
    Output("pie", "figure"),
    [Input("day-slider", "value")],
)
def update_pie(selected_day):
    """Display labels price chart."""
    
    dff = labeled_df[selected_day[0]:selected_day[1]]
    
    # pie chart data
    cnt = dff.label.value_counts()
    labels = list(cnt.index)
    values = list(cnt.values)

    piedata = go.Pie(labels=labels, 
                     values=values, 
                     hole=0.3,
                     sort=False,
                     marker={'colors': ['#0033ff', '#33ff00', '#ff0000']})
    
    return {"data": [piedata], 
            "layout": go.Layout(
                    height=150,
                    plot_bgcolor=app_color["graph_bg"],
                    paper_bgcolor=app_color["graph_bg"],
                    font={'color': '#fff'},
                    margin={'t': 5, 'b': 5},
                    autosize=True,
                    #legend_orientation="h"
                )
            }

if __name__ == '__main__':
    app.run_server(debug=False)


