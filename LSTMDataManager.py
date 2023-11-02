import copy
import os.path
import sys
from dateutil.relativedelta import relativedelta
from SequenceDataset import SequenceDataset
from torch.utils.data import DataLoader
import tqdm
from utils import *
from matplotlib import pyplot as plt

torch.manual_seed(2055)


class LSTMDataManager:
    def __init__(self, full_data_path):
        self.full_data_path = full_data_path
        self.forcast_lead_hours = 48
        self.learning_rate = 1e-6
        self.num_layers = 3
        self.num_hidden_units = 12
        self.train_test_ratio = 0.7
        self.epochs = 5
        self.batch_size = 7
        self.seq_length = 2 * self.forcast_lead_hours * 2
        self.target_variable = 'hs'
        self.output_path = "./outputs"

        self.model_path = f"{self.output_path}/shallow_reg_lstm_{self.forcast_lead_hours}h.pth"
        self.train_loader_path = f"{self.output_path}/train_loader_{self.forcast_lead_hours}h_forcast.pth"
        self.test_loader_path = f"{self.output_path}/test_loader_{self.forcast_lead_hours}h_forcast.pth"
        self.train_eval_loader_path = f"{self.output_path}/train_eval_loader_{self.forcast_lead_hours}h_forcast.pth"
        self.new_data_csv_path = r".\datasets\cameri_buoy_data\haifa_new_data.csv"
        self.model = torch.load(self.model_path) if os.path.exists(self.model_path) else None
        self.train_loader = torch.load(self.train_loader_path) if os.path.exists(self.model_path) else None
        self.test_loader = torch.load(self.test_loader_path) if os.path.exists(self.model_path) else None
        self.train_eval_loader = torch.load(self.train_eval_loader_path) if os.path.exists(self.model_path) else None
        self.loss_function = None
        self.optimizer = None
        self.features = None

    def build_new_model(self):
        # Load data
        print(f"###########\t\tCreating model for forcast lead:  {self.forcast_lead_hours} hours")

        forecast_lead = self.forcast_lead_hours * 2  # Since rows are for each 30 min, not 1 hour.
        full_data_df, self.features, new_target = load_data(csv_path=self.full_data_path,
                                                            features_mask='all',
                                                            forecast_lead=forecast_lead,
                                                            target_variable=self.target_variable)
        df_train, df_test = train_test_split(full_data_df, ratio=self.train_test_ratio)

        train_dataset = SequenceDataset(df_train, new_target, self.features, self.seq_length)
        test_dataset = SequenceDataset(df_test, new_target, self.features, self.seq_length)
        train_dataset.normalize_features_and_target()
        test_dataset.columns_std = copy.deepcopy(train_dataset.columns_std)
        test_dataset.columns_mean = copy.deepcopy(train_dataset.columns_mean)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.train_eval_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        torch.save(self.train_loader, self.train_loader_path)
        torch.save(self.test_loader, self.test_loader_path)
        torch.save(self.train_eval_loader, self.train_eval_loader_path)

        # train model
        self.model, self.loss_function, self.optimizer = configure_new_model(self.features, self.learning_rate,
                                                                             self.num_hidden_units,self.num_layers)
        self.train_lstm()

    def run_existing_model(self, is_plot_predictions=False):
        if None in [self.model, self.train_loader, self.test_loader, self.train_eval_loader]:
            print("ERROR: one of the files was not found. exiting")
            print(f"model={self.model}\ntrain_loader={self.train_loader}\ntest_loader={self.test_loader}\n"
                  f"train_eval_loader={self.train_eval_loader}")
            return
        pred_value_col_name = "Model forecast"

        df_train, df_test = self.train_loader.dataset.df, self.test_loader.dataset.df
        columns_mean = self.train_loader.dataset.columns_mean
        columns_std = self.train_loader.dataset.columns_std

        df_train[pred_value_col_name] = self.model.predict(self.train_eval_loader).numpy()
        df_test[pred_value_col_name] = self.model.predict(self.test_loader).numpy()

        df_out = pd.concat((df_train, df_test))[[self.target_variable, pred_value_col_name]]

        for c in df_out.columns:
            df_out[c] = df_out[c] * columns_std[c] + columns_mean[c]

        df_out.to_csv(f"{self.output_path}/predictions_output_{self.forcast_lead_hours}h.csv")
        if is_plot_predictions:
            plot_predictions(df_out, df_test.index[0], self.target_variable)

    def predict_on_new_data_csv(self, data_to_predict_csv_path, is_plot_predictions=False):

        train_loader = self.train_loader
        mean_std_dicts = (train_loader.dataset.columns_mean, train_loader.dataset.columns_std)
        target_mean, target_std = mean_std_dicts[0][self.target_variable], mean_std_dicts[1][self.target_variable]
        self.features = train_loader.dataset.features
        # Need to put the new data in a loader.
        prediction_df, _, new_target = load_data(csv_path=data_to_predict_csv_path,
                                                 features_mask=self.features,
                                                 forecast_lead=self.forcast_lead_hours * 2,
                                                 target_variable=self.target_variable,
                                                 new_data=True)
        new_dataset = SequenceDataset(prediction_df, new_target, self.features, self.seq_length)

        new_dataset.normalize_features_and_target(mean_std_dicts)
        new_loader = DataLoader(new_dataset, batch_size=self.batch_size, shuffle=False)

        predictions_column_name = 'Model Prediction'
        prediction_df[predictions_column_name] = self.model.predict(new_loader) * target_std + target_mean
        prediction_df[self.target_variable] = prediction_df[self.target_variable]
        prediction_df = prediction_df.loc[:, [self.target_variable, predictions_column_name]]
        forcast_start_datetime = prediction_df.index[-1 * self.forcast_lead_hours * 2]
        predicted_values_only = prediction_df.loc[forcast_start_datetime:, predictions_column_name]
        predicted_values_only = predicted_values_only.reset_index()
        predicted_values_only['datetime'] = pd.to_datetime(predicted_values_only['datetime'])
        predicted_values_only['datetime'] = predicted_values_only['datetime'].apply(lambda x: x + relativedelta(
            hours=self.forcast_lead_hours))

        prediction_df.to_csv(f"{self.output_path}/new_data_predictions_full.csv")
        predicted_values_only.to_csv(f"{self.output_path}/new_data_predictions_forcast_only.csv", index=False)
        # compare wuth real results
        empirical_measurments_path = r".\outputs\new_data_real_measurments.csv"
        empirical_measurments = pd.read_csv(empirical_measurments_path, index_col="datetime")
        empirical_measurments[predictions_column_name] = predicted_values_only.loc[:len(empirical_measurments)-1][predictions_column_name].values
        f, ax = plt.subplots(1, 1)
        ax.plot(range(len(empirical_measurments)), empirical_measurments[predictions_column_name])
        ax.plot(range(len(empirical_measurments)), empirical_measurments[self.target_variable])
        ax.set_title('Emp. data vs predicted data')
        # ax.legend()
        plt.show()
        if is_plot_predictions:
            plot_predictions(prediction_df, prediction_df.index[0], self.target_variable)


    def train_lstm(self):
        print("Untrained test\n--------")
        self.model.test_model(self.test_loader, self.loss_function)
        train_loss = []
        test_loss = []
        for ix_epoch in tqdm.tqdm(range(self.epochs)):
            print(f"Epoch {ix_epoch}\n---------")
            self.model, train_loss_epoch = self.model.train_epoch(self.train_loader, self.loss_function, self.optimizer)
            test_loss_epoch = self.model.test_model(self.test_loader, self.loss_function)
            torch.save(self.model, self.model_path)
            self.predict_on_new_data_csv(self.new_data_csv_path, False)
            if ix_epoch == 0:
                continue
            train_loss.append(train_loss_epoch)
            test_loss.append(test_loss_epoch)
        f, ax = plt.subplots(1, 1)
        ax.plot(range(self.epochs-1), train_loss)
        ax.plot(range(self.epochs-1), test_loss)
        ax.set_title('Train/Test Loss')
        ax.legend()
        plt.show()
