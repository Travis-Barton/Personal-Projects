import chess.pgn
import pandas as pd
# pgn = open('/Users/tbarton/Documents/GitHub/Personal-Projects/chess_bot/OM TOP OPENINGS/RuyLopez/RuyLopez.pgn')
chess_file = '/Users/tbarton/Documents/GitHub/Personal-Projects/chess_bot/OM TOP OPENINGS/Scandinavian/Scandinavian_3_Qd6.pgn'


def create_chess_dataframe(chess_file):
    pgn = open(chess_file, encoding='iso-8859-1')
    games_data = pd.DataFrame(columns=['Result', 'PlyCount', 'GameText'])
    game = chess.pgn.read_game(pgn)

    while game is not None:
        try:
            if game.headers['Result'] == '1/2-1/2':
                continue
            else:
                games_data = games_data.append({'Result': game.headers['Result'],
                                                'PlyCount': game.headers['PlyCount'],
                                                'GameText': str(game.mainline_moves())}, ignore_index=True)

                # Start collecting the important info
        except Exception as e:
            print(e)
        game = chess.pgn.read_game(pgn)

    return games_data

str(game.mainline())
game.headers['Result']
