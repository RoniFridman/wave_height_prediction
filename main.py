import copy
import os.path

from LSTM import LSTM
from train_model import train
from utils import *
from SequenceDataset import SequenceDataset
from ShallowRegressionLSTM import ShallowRegressionLSTM
from torch.utils.data import DataLoader
from train import train_lstm
from test import test_model
from predict import predict

torch.manual_seed(99)
learning_rate = 1e-5
num_hidden_units = 16
epochs = 10


def build_new_model(csv_path, seq_length, forecast_lead, target_variable):
    # Load data
    batch_size = 4

    training_set, features, target_variable, target = load_data(csv_path=csv_path,
                                                                forecast_lead=forecast_lead,
                                                                target_variable=target_variable)
    df_train, df_test = train_test_split(training_set)
    df_train, df_test, target_mean, target_stdv = normalize_features_and_target(df_train, df_test, target)

    train_dataset = SequenceDataset(df_train, target=target, features=features, sequence_length=seq_length,
                                    target_mean=target_mean, target_stdv=target_stdv)
    test_dataset = SequenceDataset(df_test, target=target, features=features, sequence_length=seq_length,
                                   target_mean=target_mean, target_stdv=target_stdv)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_loader_path = f"./train_loader_{hours_to_predict}h_forcast.pth"
    test_loader_path = f"./test_loader_{hours_to_predict}h_forcast.pth"
    train_eval_loader_path = f"./train_eval_loader_{hours_to_predict}h_forcast.pth"

    torch.save(train_loader, train_loader_path)
    torch.save(test_loader, test_loader_path)
    torch.save(train_eval_loader, train_eval_loader_path)

    # train model

    model, loss_function, optimizer = configure_model(features, learning_rate, num_hidden_units, model_path=None)
    train_lstm(model, loss_function, optimizer, train_loader, test_loader, forecast_lead, epochs)


def run_existing_model(plot_predictions=False):
    target_variable = 'hs'
    hours_to_predict = 5
    model_path = f"./sr_lstm_model_{hours_to_predict}h_forcast.pth"
    train_loader_path = f"./train_loader_{hours_to_predict}h_forcast.pth"
    test_loader_path = f"./test_loader_{hours_to_predict}h_forcast.pth"
    train_eval_loader_path = f"./train_eval_loader_{hours_to_predict}h_forcast.pth"

    if os.path.exists(model_path) and os.path.exists(train_loader_path) \
            and os.path.exists(test_loader_path) and os.path.exists(train_eval_loader_path):
        model, train_loader, test_loader, train_eval_loader = load_model_and_loaders(model_path, train_loader_path,
                                                                                     test_loader_path,
                                                                                     train_eval_loader_path)
        ystar_col = "Model forecast"

        df_train, df_test = train_loader.dataset.df, test_loader.dataset.df
        target_stdev, target_mean = train_loader.dataset.target_variable_mean_stdv

        df_train[ystar_col] = predict(train_eval_loader, model).numpy()
        df_test[ystar_col] = predict(test_loader, model).numpy()

        df_out = pd.concat((df_train, df_test))[[target_variable, ystar_col]]

        for c in df_out.columns:
            df_out[c] = df_out[c] * target_stdev + target_mean

        df_out.to_csv(f"./predictions_output.csv")
        if plot_predictions:
            plot_predictions(df_out, df_test.index[0], target_variable)
    else:
        print("ERROR: one of the files was not found. exiting")


if __name__ == "__main__":
    data_csv_path = r"C:\Users\soldier109\PycharmProjects\CAMERI_Work\wave_height_prediction\datasets\cameri_buoy_data\haifa_short.csv"
    target_variable = 'hs'
    hours_to_predict = 5
    seq_length = 100
    forecast_lead = hours_to_predict * 2  # since every row is 30 min
    predict_using_existing_model = True

    if predict_using_existing_model:
        run_existing_model(plot_predictions=True)
    else:
        build_new_model(csv_path=data_csv_path, seq_length=seq_length, forecast_lead=forecast_lead,
                        target_variable=target_variable)
        run_existing_model()
