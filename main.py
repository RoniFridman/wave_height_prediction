from LSTMDataManager import LSTMDataManager
import sys
from utils import update_latest_data_from_db
import os

if __name__ == "__main__":
    location, target_variable, function = sys.argv[1], sys.argv[2], sys.argv[3]
    # target_variable = 'hs'
    # location = 'haifa'
    # function = 'build_new_model'


    # base_location_folder = "/data/scripts/waves_height_prediction"
    base_location_folder = "."

    full_data_path = os.path.join(base_location_folder,'datasets','cameri_buoy_data','haifa.csv')
    lstm_data_manager = LSTMDataManager(full_data_path,location,target_variable, base_location_folder)
    if function == 'build_new_model':
        lstm_data_manager.build_new_model()
    elif function == 'predict_latest_available_data':
        update_latest_data_from_db(full_data_path)
        lstm_data_manager.predict_latest_available_data(location)
    elif function == 'build_predict_upload':
        lstm_data_manager.build_new_model()
        update_latest_data_from_db(full_data_path)
        lstm_data_manager.predict_latest_available_data(location)