import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy
from ShallowRegressionLSTM import ShallowRegressionLSTM
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

def load_data(data, features_mask='all', forecast_lead=4, target_variable='hs', new_data=False):
    if isinstance(data, str):
        data = pd.read_csv(data, index_col='datetime')
    if "id" in data.columns:
        data.drop(columns=["id"], inplace=True)
    if "location_id" in data.columns:
        data.drop(columns=["location_id"],inplace=True)
    data['direction_x'] = data['direction'].apply(lambda x: np.cos(x / 360))
    data['direction_y'] = data['direction'].apply(lambda x: np.sin(x / 360))
    data.drop(columns=['direction'], inplace=True)
    if features_mask != 'all':
        features_mask = list(features_mask) + [target_variable]
        data = data.loc[:, features_mask]
    features = list(data.columns.difference([target_variable]))
    new_target_col_name = f"{target_variable}_lead{forecast_lead}"

    if not new_data:
        data[new_target_col_name] = data[target_variable].shift(-forecast_lead)
        data = data.iloc[:-forecast_lead]
    else:
        data[new_target_col_name] = data[target_variable].shift(-forecast_lead)

    for column in features + [new_target_col_name]:
        iqr = scipy.stats.iqr(data[column], nan_policy='omit')
        q1, q3 = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)
        lb, ub = q1 - 2 * iqr, q3 + 2 * iqr
        data.loc[(data[column] < lb) | (data[column] > ub), column] = np.nan
        data[column].interpolate(inplace=True)
        data[column].fillna(method='bfill', inplace=True)
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


def configure_new_model(features, learning_rate, num_hidden_units, num_layers):
    model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units, num_layers=num_layers)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_function, optimizer


def create_short_data_csv(full_csv_path, new_data_training_path, empirical_test_data_path,
                          forecast, seq_number, predict_latest=False, years_of_data_to_use=3):
    if predict_latest:
        data = pd.read_csv(full_csv_path).set_index('datetime').iloc[-1 * forecast * seq_number:]
        return data
    cutoff_index= round(-17520 * years_of_data_to_use)
    data = pd.read_csv(full_csv_path).set_index('datetime').iloc[cutoff_index:]
    data[-1 * forecast * seq_number:-1 * forecast].to_csv(new_data_training_path)
    data[-1 * forecast:].to_csv(empirical_test_data_path)
    data = data.iloc[:-1 * forecast]
    return data

def upload_predictions_to_db(predictions_df: pd.DataFrame, location='haifa',target_variable='hs'):
    """
    saves the prediction made into a dedicated table
    order in table should be: datetime, "predicted_<target variable name>"
    """
    try:

        connection = psycopg2.connect(user=os.getenv('db_user'),
                                      password=os.getenv('db_password'),
                                      host=os.getenv('db_host'),
                                      port=os.getenv('db_port'),
                                      database=os.getenv('db_name'))
        table_name = f"waves_data.cameri_{target_variable}_predictions_{location}"
        target_variable_table_name = f"predicted_{target_variable}"
        cursor = connection.cursor()
        general_command = "INSERT INTO " + table_name + " VALUES (%s, %s) on conflict (datetime) do update set "+\
                f"{target_variable_table_name} = excluded.{target_variable_table_name}"
        for l in predictions_df:
            command = cursor.mogrify(general_command, (l[0],l[1])).decode("utf-8")
            cursor.execute(command)
        connection.commit()
    except (Exception, psycopg2.Error) as error:
        print(" saving to db is not successful : " + str(error))

    finally:
        if connection:
            cursor.close()
            connection.close()


def update_latest_data_from_db(full_data_path, location='haifa'):
    """
    saves the prediction made into a dedicated table
    order in table should be: datetime, location, target variable predicted value
    """
    try:
        if os.path.exists(full_data_path):
            os.remove(full_data_path)

        connection = psycopg2.connect(user=os.getenv('db_user'),
                                      password=os.getenv('db_password'),
                                      host=os.getenv('db_host'),
                                      port=os.getenv('db_port'),
                                      database=os.getenv('db_name'))
        location_id = {'haifa':1, "ashdod":2}

        command = f"COPY (select datetime,hs,direction,tz,tp,temperature,h_onethird,hmax,tav " \
                  f"from backend_buoysmsr where location_id = {location_id[location]} order by datetime) " \
                  f"to STDOUT with CSV delimiter ',' header"

        cursor = connection.cursor()
        command = cursor.mogrify(command)
        connection.commit()
        with open(full_data_path,"w+") as file:
            cursor.copy_expert(command, file)
            file.close()

    except (Exception, psycopg2.Error) as error:
        print(" exporting to CSV was not successful : " + str(error))

    finally:
        if connection:
            cursor.close()
            connection.close()