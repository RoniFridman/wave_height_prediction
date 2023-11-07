from LSTMDataManager import LSTMDataManager
import sys
from utils import update_latest_data_from_db
import os
from dotenv import load_dotenv
import pathlib
load_dotenv()

if __name__ == "__main__":
    py_file_path, function = sys.argv[0], sys.argv[1]
    base_location_folder = str(pathlib.Path(sys.argv[0]).parent)
    location, target_variable = os.getenv("LOCATION"), os.getenv("TARGET_VARIABLE")
    os.makedirs(os.path.join(base_location_folder, "text_outputs"), exist_ok=True)
    full_data_path = os.path.join(base_location_folder, 'datasets', 'cameri_buoy_data', f'{location}.csv')
    wind_data_path = os.path.join(base_location_folder, 'datasets', 'cameri_buoy_data', f'{location}_wind.csv')
    lstm_data_manager = LSTMDataManager(full_data_path, wind_data_path, location, target_variable, base_location_folder)

    if function == 'build':
        lstm_data_manager.build_new_model()
    elif function == "update_data":
        update_latest_data_from_db(full_data_path)
    elif function == 'predict_latest':
        # update_latest_data_from_db(full_data_path)
        lstm_data_manager.predict_latest_available_data(location)
    elif function == 'build_predict_upload':
        lstm_data_manager.build_new_model()
        update_latest_data_from_db(full_data_path)
        lstm_data_manager.predict_latest_available_data(location)
