import copy
import os.path
import sys


from SequenceDataset import SequenceDataset
from torch.utils.data import DataLoader
import tqdm
from utils import *
from matplotlib import pyplot as plt

torch.manual_seed(99)


class LSTMDataManager:
    def __init__(self, full_data_path):
        self.full_data_path = full_data_path
        self.learning_rate = 1e-5
        self.num_hidden_units = 16
        self.train_test_ratio = 0.8  # train is 90%
        self.epochs = 30
        self.batch_size = 4
        self.seq_length = 200
        self.target_variable = 'hs'
        self.output_path = "./outputs"
        self.forcast_lead_hours = 48
        self.model_path = f"{self.output_path}/shallow_reg_lstm_{self.forcast_lead_hours}h.pth"
        self.train_loader_path = f"{self.output_path}/train_loader_{self.forcast_lead_hours}_forcast.pth"
        self.test_loader_path = f"{self.output_path}/test_loader_{self.forcast_lead_hours}_forcast.pth"
        self.train_eval_loader_path = f"{self.output_path}/train_eval_loader_{self.forcast_lead_hours}_forcast.pth"

        self.model = torch.load(self.model_path) if os.path.exists(self.model_path) else None
        self.train_loader = torch.load(self.train_loader_path) if os.path.exists(self.model_path) else None
        self.test_loader = torch.load(self.test_loader_path) if os.path.exists(self.model_path) else None
        self.train_eval_loader = torch.load(self.train_eval_loader_path) if os.path.exists(self.model_path) else None
        self.loss_function = None
        self.optimizer = None
        self.features = None
        self.target_mean = 0
        self.target_std = 0

    def build_new_model(self):
        # Load data
        print(f"###########\t\tCreating model for forcast lead:  {self.forcast_lead_hours}")

        forecast_lead = self.forcast_lead_hours * 2  # Since rows are for each 30 min, not 1 hour.
        full_data_df, self.features, new_target = load_data(csv_path=self.full_data_path,
                                                            forecast_lead=forecast_lead,
                                                            target_variable=self.target_variable)
        df_train, df_test = train_test_split(full_data_df, ratio=self.train_test_ratio)
        df_train, df_test, self.target_mean, self.target_std = normalize_features_and_target(df_train, df_test,
                                                                                             new_target)

        train_dataset = SequenceDataset(df_train, new_target, self.features, self.seq_length)
        test_dataset = SequenceDataset(df_test, new_target, self.features, self.seq_length)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.train_eval_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        torch.save(self.train_loader, self.train_loader_path)
        torch.save(self.test_loader, self.test_loader_path)
        torch.save(self.train_eval_loader, self.train_eval_loader_path)

        # train model
        self.model, self.loss_function, self.optimizer = configure_new_model(self.features, self.learning_rate,
                                                                             self.num_hidden_units)
        self.train_lstm()

    def run_existing_model(self, is_plot_predictions=False):
        if None in [self.model, self.train_loader, self.test_loader, self.train_eval_loader]:
            print("ERROR: one of the files was not found. exiting")
            print(f"model={self.model}\ntrain_loader={self.train_loader}\ntest_loader={self.test_loader}\n"
                  f"train_eval_loader={self.train_eval_loader}")
            return
        pred_value_col_name = "Model forecast"

        df_train, df_test = self.train_loader.dataset.df, self.test_loader.dataset.df
        df_train[pred_value_col_name] = self.model.predict( self.train_eval_loader).numpy()
        df_test[pred_value_col_name] = self.model.predict(self.test_loader).numpy()

        df_out = pd.concat((df_train, df_test))[[self.target_variable, pred_value_col_name]]

        for c in df_out.columns:
            df_out[c] = df_out[c] * self.target_std + self.target_mean

        df_out.to_csv(f"{self.output_path}/predictions_output_{self.forcast_lead_hours}h.csv")
        if is_plot_predictions:
            plot_predictions(df_out, df_test.index[0], self.target_variable)

    def predict_new_data(self, data_to_predict, is_plot_predictions=False):
        predictions_df = self.train_loader.dataset.df
        predictions = []
        output: torch.Tensor = self.model.predict(self.model, self.train_loader) * self.target_std + self.target_mean
        for out in output:
            predictions.append([self.forcast_lead_hours, float(out)])
        temp = pd.DataFrame(predictions, columns=["lead_time", "pred_value"])
        if len(predictions_df) > len(temp):
            predictions_df = self.train_loader.dataset.df[-1 * len(temp):]
        predictions_df[self.target_variable] = predictions_df[self.target_variable] * self.target_std + self.target_mean
        predictions_df[f'pred_value_{self.forcast_lead_hours}'] = temp['pred_value'].values

        predictions_df.to_csv(f"{self.output_path}/new_data_predictions.csv")
        if is_plot_predictions:
            plot_predictions(predictions_df, predictions_df.index[0], self.target_variable)

    def train_lstm(self):
        print("Untrained test\n--------")
        self.model.test_model(self.test_loader, self.loss_function)
        train_loss = []
        test_loss = []
        for ix_epoch in tqdm.tqdm(range(self.epochs)):
            print(f"Epoch {ix_epoch}\n---------")
            self.model, loss = self.model.train_model(self.train_loader, self.loss_function, self.optimizer)
            train_loss.append(loss)
            test_loss.append(self.model.test_model(self.test_loader, self.loss_function))

        plt.plot([[x,y] for x,y in enumerate(train_loss)])
        plt.title("Train Loss")
        plt.plot([[x, y] for x, y in enumerate(test_loss)])
        plt.title("Test Loss")
        plt.show()
        torch.save(self.model, self.model_path)
