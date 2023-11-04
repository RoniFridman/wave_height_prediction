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
        self.forcast_lead_hours = 24
        self.full_data_path = full_data_path

        self.output_path = f"./outputs/{os.path.basename(full_data_path).split('.')[0]}_{self.forcast_lead_hours}h_forecast"
        self.images_output_path = os.path.join(self.output_path, "images")
        self.csv_output_path = os.path.join(self.output_path, "csv files")
        os.makedirs(self.images_output_path, exist_ok=True)
        os.makedirs(self.csv_output_path, exist_ok=True)

        self.new_data_training_path = f"{self.csv_output_path}\haifa_data_for_forcast_{self.forcast_lead_hours}h.csv"
        self.forecast_groundtruth = f"{self.csv_output_path}\haifa_groundtruth_{self.forcast_lead_hours}h.csv"
        self.learning_rate = 5e-4
        self.num_layers = 3
        self.num_hidden_units = 64
        self.train_test_ratio = 0.8
        self.epochs = 50
        self.batch_size = 4
        # self.seq_length = self.forcast_lead_hours * 1
        self.seq_length = self.forcast_lead_hours // 2
        self.target_variable = 'hs'


        self.model_path = f"{self.output_path}/shallow_reg_lstm_{self.forcast_lead_hours}h.pth"
        self.train_loader_path = f"{self.output_path}/train_loader_{self.forcast_lead_hours}h_forcast.pth"
        self.test_loader_path = f"{self.output_path}/test_loader_{self.forcast_lead_hours}h_forcast.pth"
        self.train_eval_loader_path = f"{self.output_path}/train_eval_loader_{self.forcast_lead_hours}h_forcast.pth"

        self.model = torch.load(self.model_path) if os.path.exists(self.model_path) else None
        self.train_loader = torch.load(self.train_loader_path) if os.path.exists(self.model_path) else None
        self.test_loader = torch.load(self.test_loader_path) if os.path.exists(self.model_path) else None
        self.train_eval_loader = torch.load(self.train_eval_loader_path) if os.path.exists(self.model_path) else None
        self.loss_function = None
        self.optimizer = None
        self.features = None
        self.training_counter=0

    def build_new_model(self):
        # Load data
        print(f"###########\t\tCreating model for forcast lead:  {self.forcast_lead_hours} hours")

        forecast_lead = self.forcast_lead_hours * 2  # Since rows are for each 30 min, not 1 hour.
        self.full_data_df = create_short_data_csv(self.full_data_path, self.new_data_training_path, self.forecast_groundtruth, forecast_lead, self.seq_length)
        full_data_df, self.features, new_target = load_data(data=self.full_data_df,
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

    def predict_on_new_data_csv(self, data_to_predict_csv_path):
        train_loader = self.train_loader
        mean_std_dicts = (train_loader.dataset.columns_mean, train_loader.dataset.columns_std)
        target_mean, target_std = mean_std_dicts[0][self.target_variable], mean_std_dicts[1][self.target_variable]
        self.features = train_loader.dataset.features
        # Need to put the new data in a loader.
        prediction_df, _, new_target = load_data(data=data_to_predict_csv_path,
                                                 features_mask=self.features,
                                                 forecast_lead=self.forcast_lead_hours * 2,
                                                 target_variable=self.target_variable,
                                                 new_data=True)
        new_dataset = SequenceDataset(prediction_df, new_target, self.features, self.seq_length)
        new_dataset.normalize_features_and_target(mean_std_dicts)
        new_loader = DataLoader(new_dataset, batch_size=self.batch_size, shuffle=False)

        predictions_column_name = 'Model Prediction'
        prediction_df[predictions_column_name] = self.model.predict(new_loader) * target_std + target_mean
        prediction_df[self.target_variable] = prediction_df[self.target_variable] * target_std + target_mean
        prediction_df = prediction_df.loc[:, [self.target_variable, predictions_column_name]]
        forcast_start_datetime = prediction_df.index[-1 * self.forcast_lead_hours * 2]
        predicted_values_only = prediction_df.loc[forcast_start_datetime:, predictions_column_name]
        predicted_values_only = predicted_values_only.reset_index()
        predicted_values_only['datetime'] = pd.to_datetime(predicted_values_only['datetime'])
        predicted_values_only['datetime'] = predicted_values_only['datetime'].apply(lambda x: x + relativedelta(
            hours=self.forcast_lead_hours))

        prediction_df.to_csv(f"{self.csv_output_path}/new_data_predictions_full.csv")
        predicted_values_only.to_csv(f"{self.csv_output_path}/new_data_predictions_forcast_only.csv", index=False)
        # compare with real results

        empirical_measurments = pd.read_csv(self.forecast_groundtruth, index_col="datetime")[:len(predicted_values_only)]
        empirical_measurments[predictions_column_name] = predicted_values_only.loc[:len(empirical_measurments)][predictions_column_name].values
        plt.plot(range(len(empirical_measurments)), empirical_measurments[predictions_column_name], label="Predictions")
        plt.plot(range(len(empirical_measurments)), empirical_measurments[self.target_variable], label="Real Data")
        plt.title(f'{self.training_counter} : Forecast Predictions vs Real Data epoch {self.training_counter}')
        plt.legend()
        output_image_path = os.path.join(self.images_output_path, f"forecast_predictions_epoch_{self.training_counter}.png")
        plt.savefig(output_image_path)
        plt.close()

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
            self.predict_on_new_data_csv(self.new_data_training_path)
            self.training_counter += 1
            if ix_epoch == 0:
                continue
            train_loss.append(train_loss_epoch)
            test_loss.append(test_loss_epoch)
        plt.plot(range(self.epochs-1), train_loss, label="Train Loss")
        plt.plot(range(self.epochs-1), test_loss,label="Test Loss")
        plt.title('Train/Test Loss')
        plt.legend()
        plt.savefig(os.path.join(self.images_output_path,f"loss_graphs.png"))
        plt.close()