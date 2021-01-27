import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import chess

chess_data = pd.read_csv('/Users/tbarton/Documents/GitHub/Personal-Projects/chess_bot/games.csv')
chess_data.shape
chess_data.columns

#%% clean the data
mates = chess_data.query('victory_status == "mate"')
chess_data.query('(victory_status == "resign") & (rated == True)').turns.hist(bins=50)
plt.show()
chess_data.query('(victory_status == "resign") & (rated == True)').white_rating.hist(bins=50)
plt.show()
chess_data.query('(victory_status == "resign") & (rated == True)').black_rating.hist(bins=50)
plt.show()
temp = chess_data.query('(victory_status == "resign") & (rated == True)')
temp.loc[:, 'rating_diff'] = temp.white_rating - temp.black_rating
temp.rating_diff.abs().hist(bins=50)
plt.show()
chess_data['rating_diff'] = np.abs(chess_data.white_rating - chess_data.black_rating)
resigns = chess_data.query("(white_rating > 1100) & (black_rating > 1100) & (rating_diff < 300) & " +
                           "(victory_status == 'resign') & (turns > 5)")
resigns.turns.hist(bins=50); plt.show()
resigns.turns.describe()

full_dataframe = pd.concat([mates, resigns], axis=0)
full_dataframe['rating_diff'] = np.abs(full_dataframe.white_rating - full_dataframe.black_rating)

#%% create the text file

