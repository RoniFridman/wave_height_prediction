from LSTMDataManager import LSTMDataManager



if __name__ == "__main__":
    full_data_path = r"C:\Users\soldier109\PycharmProjects\CAMERI_Work\waves_height_prediction\datasets\cameri_buoy_data\haifa_short.csv"
    new_data_csv_path = r"C:\Users\soldier109\PycharmProjects\CAMERI_Work\waves_height_prediction\datasets\cameri_buoy_data\haifa_new_data.csv"
    target_variable = 'hs'
    lstm_data_manager = LSTMDataManager(full_data_path)