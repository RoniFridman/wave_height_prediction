from LSTMDataManager import LSTMDataManager


if __name__ == "__main__":
    full_data_path = r".\datasets\cameri_buoy_data\haifa_short.csv"
    new_data_csv_path = r".\datasets\cameri_buoy_data\haifa_new_data.csv"
    lstm_data_manager = LSTMDataManager(full_data_path)
    lstm_data_manager.build_new_model()
    # lstm_data_manager.run_existing_model(is_plot_predictions=True)
    # lstm_data_manager.predict_on_new_data_csv(new_data_csv_path, False)