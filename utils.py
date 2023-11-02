import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy
from ShallowRegressionLSTM import ShallowRegressionLSTM
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def load_model_and_loaders(model_path, train_loader_path, test_loader_path, train_eval_loader_path):
    model = torch.load(model_path)
    train_loader = torch.load(train_loader_path)
    test_loader = torch.load(test_loader_path)
    train_eval_loader = torch.load(train_eval_loader_path)
    return model, train_loader, test_loader, train_eval_loader


def load_data(csv_path, features_mask='all', forecast_lead=4, target_variable='hs', new_data=False):
    data = pd.read_csv(csv_path, index_col='datetime')
    data['direction_x'] = data['direction'].apply(lambda x: np.cos(x/360))
    data['direction_y'] = data['direction'].apply(lambda x: np.sin(x/360))
    data.drop(columns=['direction'], inplace=True)
    if features_mask != 'all':
        features_mask = list(features_mask) + [target_variable]
        data = data.loc[:, features_mask]
    features = list(data.columns.difference([target_variable]))
    new_target_col_name = f"{target_variable}_lead{forecast_lead}"

    if not new_data:
        data[new_target_col_name] = data[target_variable].shift(-forecast_lead)
        data = data.iloc[:-forecast_lead]

    for column in features:
        iqr = scipy.stats.iqr(data[column])
        median = np.median(data[column])
        lb, ub = median - iqr, median + iqr
        data.loc[(data[column] < lb) | (data[column] > ub),column] = np.nan
        data[column].interpolate(inplace=True)

    return data, features, new_target_col_name


def train_test_split(data, ratio=0.7, test_start_ts=None):
    splitting_edge = int(len(data) * ratio) if test_start_ts is None else test_start_ts

    if test_start_ts is not None:
        df_train = data.loc[:test_start_ts].copy()
        df_test = data.loc[test_start_ts:].copy()
    else:
        df_train = data.iloc[:splitting_edge].copy()
        df_test = data.iloc[splitting_edge:].copy()
    return df_train, df_test


def configure_new_model(features, learning_rate, num_hidden_units,num_layers):
    model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units,num_layers=num_layers)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_function, optimizer


def load_model(model_path='./lstm_model.pth'):
    model = torch.load(model_path)
    return model


def plot_predictions(df_out, test_start, target_variable: str):
    plot_template = dict(
        layout=go.Layout({
            "font_size": 18,
            "xaxis_title_font_size": 24,
            "yaxis_title_font_size": 24})
    )
    fig = px.line(df_out, labels=dict(datetime="Datetime", value=f"{target_variable.capitalize()} [m]"))
    fig.add_vline(x=test_start, line_width=4, line_dash="dash")
    fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
    fig.update_layout(
        template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
    )
    fig.show()
