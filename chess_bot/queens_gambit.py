import chess.pgn
import pandas as pd
import os
# pgn = open('/Users/tbarton/Documents/GitHub/Personal-Projects/chess_bot/OM TOP OPENINGS/RuyLopez/RuyLopez.pgn')
# chess_file = '/Users/tbarton/Documents/GitHub/Personal-Projects/chess_bot/OM TOP OPENINGS/Scandinavian/Scandinavian_3_Qd6.pgn'

# chess_text = '#Start '


def create_chess_dataframe(chess_file, encoding='iso-8859-1'):
    pgn = open(chess_file, encoding=encoding)
    games_data = pd.DataFrame(columns=['Result', 'PlyCount', 'GameText'])

    game = 'not_none'

    while game is not None:
        try:
            # if game.headers['Result'] == '1/2-1/2':
            #     games_data_draw = games_data_draw.append({'Result': game.headers['Result'],
            #                                     'PlyCount': game.headers['PlyCount'],
            #                                     'GameText': str(game.mainline_moves())}, ignore_index=True)
            # else:
            #     #break
            games_data = games_data.append({'Result': game.headers['Result'],
                                            'PlyCount': game.headers['PlyCount'],
                                            'GameText': str(game.mainline_moves())}, ignore_index=True)

                # Start collecting the important info
        except Exception as e:
            print(e)
        game = chess.pgn.read_game(pgn)

    return games_data

def itterate_through_files(dirpath, file_type='pgn', encoding='iso-8859-1'):
    games_data_new = pd.DataFrame()
    for folder in os.listdir(dirpath):
        folder_dir = os.path.join(dirpath, folder)
        if os.path.isdir(folder_dir):
            for file in os.listdir(os.path.join(dirpath, folder)):
                if file.endswith(file_type):
                    games_data_new = pd.concat([games_data_new, create_chess_dataframe(os.path.join(folder_dir, file),
                                                                                       encoding=encoding)],
                                               ignore_index=True)
    return games_data_new

def convert_to_txt(dataframe, query=None, save_path='', ret=False):
    if query is not None:
        dataframe = dataframe.query(query)

    ret_text = ' <|endoftext|> '.join(dataframe.loc[:, 'GameText'].tolist()) + ' <|endoftext|>'
    with open(save_path, 'w') as save:
        save.write(ret_text)
        save.close()
    if ret:
        return ret_text


games_data_new.to_pickle('first_set_of_chess_files')


if __name__=='__main__':
    chess_files = '/Users/tbarton/Documents/GitHub/Personal-Projects/chess_bot/OM TOP OPENINGS'
    chess_games_dataset = itterate_through_files(chess_files)
