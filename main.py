from LSTMDataManager import LSTMDataManager
import sys
from utils import update_latest_data_from_db
import os
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    function = sys.argv[1]
    location, target_variable = os.getenv("LOCATION"), os.getenv("TARGET_VARIABLE")
    base_location_folder = os.getenv("BASE_PROJECT_FOLDER")
    os.makedirs(os.path.join(base_location_folder,"text_outputs"), exist_ok=True)
    full_data_path = os.path.join(base_location_folder,'datasets','cameri_buoy_data',f'{location}.csv')
    wind_data_path = os.path.join(base_location_folder,'datasets','cameri_buoy_data',f'{location}_wind.csv')
    lstm_data_manager = LSTMDataManager(full_data_path,wind_data_path,location,target_variable, base_location_folder)
    if function == 'build_new_model':
        lstm_data_manager.build_new_model()
    elif function == "update_latest_data_from_db":
        update_latest_data_from_db(full_data_path)
    elif function == 'predict_latest_available_data':
        update_latest_data_from_db(full_data_path)
        lstm_data_manager.predict_latest_available_data(location)
    elif function == 'build_predict_upload':
        lstm_data_manager.build_new_model()
        update_latest_data_from_db(full_data_path)
        lstm_data_manager.predict_latest_available_data(location)