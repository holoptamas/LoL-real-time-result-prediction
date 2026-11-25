from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from LoL_live_game_prediction.Neural_Network.league_dataset import LeagueTrainDataset, LeagueTestDataset, LeagueLiveGameDataset
from LoL_live_game_prediction.Neural_Network.LeagueNeuralNetwork import LeagueNN

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = (BASE_DIR.parent/"League_Neural_Networks"/"league_model.pth")
TRAIN_CSV_PATH = (BASE_DIR.parent/"League_Match_Data"/"DS_2026_11_16_16_00_train.csv")
LOG_CSV_PATH = (BASE_DIR.parent/"League_Match_Log")

train_df = pd.read_csv(TRAIN_CSV_PATH)
log_df = pd.read_csv(LOG_CSV_PATH/"match_10.csv")

#1: 6, 2: 8, 3: 7, 4: 7, 5: 7, 6: 7, 7: 3, 8: 5, 9: 9, 10: 9

for i in range(0, 9):
    singledf = log_df.iloc[[i]]
    blue_win = singledf['BlueTeamWin'].values[0]
    game_duration = singledf['GameDuration'].values[0]
    test_singledf = singledf.drop(['BlueTeamWin'], axis=1)

    train_dataset = LeagueTrainDataset(train_df, "BlueTeamWin")
    test_dataset = LeagueTestDataset(log_df, "BlueTeamWin", train_dataset.get_min_max_params(), train_dataset.get_role_mapping())
    test_singledata = LeagueLiveGameDataset(test_singledf, train_dataset.get_min_max_params(), train_dataset.get_role_mapping())
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    my_neural_network = LeagueNN.load_model(MODEL_PATH)

    data_dict = my_neural_network.evaluate_single(test_singledata.numeric_X[0], test_singledata.champion_X[0], test_singledata.role_X[0])

    for key, item in data_dict.items():
        print(f"{key}: {item}")
    if blue_win == 1:
        print("Blue Wins")
    else:
        print("Red Wins")
    print(f"{int(game_duration / 60)}:{game_duration % 60}")
