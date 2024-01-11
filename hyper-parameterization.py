# code to running the model and saving different results
import itertools
import os, pathlib, sys
from LSTMDataManager import LSTMDataManager
from dotenv import load_dotenv,find_dotenv
import time
import shutil
# parameters = {'NUM_HIDDEN_UNITS':[3], 'LEARNING_RATE':[0.1,0.01],
#               "BATCH_SIZE":[50],"SEQ_LENGTH_DIVIDER":[1],
#               "YEARS_TO_USE":[1], "TRAIN_TEST_RATION":[0.1,0.2]}


parameters = {'NUM_HIDDEN_UNITS':[3,5,10,32,50,80,100], 'LEARNING_RATE':[0.1,0.01,0.001,0.0001],
              "BATCH_SIZE":[1,3,5,10,30,50],"SEQ_LENGTH_DIVIDER":[0.1,0.2,0.5,1,2,4],
              "YEARS_TO_USE":[1,3,5,10], "TRAIN_TEST_RATION":[0.1,0.2,0.3]}
py_file_path, function, location, target_variable = sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3]
base_location_folder = str(pathlib.Path(sys.argv[0]).parent)


os.makedirs(os.path.join(base_location_folder, "text_outputs"), exist_ok=True)
full_data_path = os.path.join(base_location_folder, 'datasets', 'cameri_buoy_data', f'{location}.csv')
wind_data_path = os.path.join(base_location_folder, 'datasets', 'cameri_buoy_data', f'{location}_wind.csv')
hyper_param_output = base_location_folder + f"/outputs/{location}_24h_{target_variable}_forecast/hyper_param/"
os.makedirs(hyper_param_output, exist_ok=True)


set_of_parameters = itertools.product(*parameters.values())

for i,params in enumerate(set_of_parameters):
    with open(".env",'w') as f:
        f.write(f"# DB ENVs\ndb_user=cameri\ndb_password=C@mer!20231975\ndb_host=127.0.0.1\ndb_port=5432\ndb_name=cameri77_db\n\n")
        f.write(f"# Main Parameters\nFORECAST_LEAD_HOURS=24\n")
        f.write(f"# Data parameters\nYEARS_TO_USE={params[4]}\nTRAIN_TEST_RATIO={params[5]}\n\n")
        f.write(f"TORCH_SEED=422\nNORMALIZATION_METHOD=mean_std  # min_max. any other method is mean-std.\n\n\n")
        f.write(f"# Model Parameters\nNUM_LAYERS=1\nNUM_HIDDEN_UNITS={params[0]}\nLEARNING_RATE={params[1]}\n")
        f.write(f"DROPOUT=0\nBATCH_SIZE={params[2]}\nEPOCHS=20\nSEQ_LENGTH_DIVIDER={params[3]}  # sequence_length = (forecast_hours * 2) // divider")
        f.close()
    load_dotenv(override=True)
    time.sleep(3)
    lstm_data_manager = LSTMDataManager(full_data_path, wind_data_path, location, target_variable, base_location_folder)
    train_loss, test_loss = lstm_data_manager.build_new_model()
    with open(os.path.join(hyper_param_output,f"test_{i}.txt"),'w+') as f:
        curr_params = open(".env").read()
        f.write(f"params:{curr_params}\n\n train loss:{train_loss}\n\ntest loss: {test_loss}")
        f.close()
        hyper_param_output = base_location_folder + f"/outputs/{location}_24h_{target_variable}_forecast/hyper_param/"

        shutil.copy(base_location_folder+f"/outputs/{location}_24h_{target_variable}_forecast/images/loss_graphs.png",
                    os.path.join(hyper_param_output,f"test_{i}_loss.png"))
