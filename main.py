from nba_api.stats.static import players
from nba_api.stats.endpoints.playergamelog import PlayerGameLog
import numpy as np
# feature extraction
columns = [
    "SEASON_ID", "Player_ID", "Game_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", 
    "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", 
    "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "VIDEO_AVAILABLE", "HOME_AWAY"
]
season = "2024-25"    # Specify the season
season_type = "Regular Season"  # Options: Regular Season, Playoffs, etc.
player_name = input("Enter the player you want to analyze: ")
player_cat = input("Enter the category you want to analyze: ")
home_away = input("Is the game home or away?: ")

def get_player_id_by_name(player_name):
    player_list = players.find_players_by_full_name(player_name)
    if player_list:
        return player_list[0]['id']  # Return the first matched player's ID
    else:
        return None
    
def get_cat_index(player_cat):
    column_index_dict = {col: idx for idx, col in enumerate(columns)}
    index = column_index_dict.get(player_cat.upper());
    return index

index = get_cat_index(player_cat)
player_id = get_player_id_by_name(player_name=player_name)

# Fetch the player's game log
player_game_log = PlayerGameLog(
    player_id=player_id,
    season=season,
    season_type_all_star=season_type
)

game_log_data = player_game_log.player_game_log.get_dict()
headers = game_log_data['headers']
games = game_log_data['data']


# Add the "HOME_AWAY" column to headers
if "HOME_AWAY" not in headers:
    headers.append("HOME_AWAY")

# Update each game row with "HOME_AWAY" value
for game in games:
    matchup = game[headers.index("MATCHUP")]
    home_away = "H" if "vs." in matchup else "A"
    game.append(home_away)  

for game in games:
    game_info = {headers[i]: game[i] for i in range(len(headers))}
    #print(game_info)


# Create the dictionary mapping column names to their index values
cat_array = [game[index] for game in game_log_data['data']]  # 25 is the index for PTS field in each game row
matchups = [game[4] for game in game_log_data['data']]
print("cat array:",cat_array)
#print(f"{player_cat} each game:", cat_array)

team_against = input("Who is the matchup?: ")
opposing_teams = [matchup.split(' @ ')[1] if ' @ ' in matchup else matchup.split(' vs. ')[1] for matchup in matchups]
#print(opposing_teams)

# pull all the games that vs NOP (ex)
def fetch_head2head(team_against):
    # Fetch the player's game log
    
    games = game_log_data['data']  # All game rows
    headers = game_log_data['headers']  # Column headers
    #print(headers)
    # Find index of relevant columns
    matchup_index = headers.index("MATCHUP")
    wl_index = headers.index("WL")
    pts_index = headers.index("PTS")  # Example: Points scored
    ast_index = headers.index("AST")  # Example: Assists
    reb_index = headers.index("REB")  # Example: Rebounds
    home_away_index = headers.index("HOME_AWAY")
    # Filter games against the specified team
    head_to_head_games = [
        game for game in games
        if team_against in (game[matchup_index].split(' @ ')[1] if ' @ ' in game[matchup_index] else game[matchup_index].split(' vs. ')[1])
    ]
    
    # Format and return the relevant stats
    head_to_head_stats = [
        {
            "GAME_DATE": game[headers.index("GAME_DATE")],
            "MATCHUP": game[matchup_index],
            "WL": game[wl_index],
            "PTS": game[pts_index],
            "AST": game[ast_index],
            "REB": game[reb_index],
            "H/A": game[home_away_index]
        }
        for game in head_to_head_games
    ]
    
    return head_to_head_stats

h2h=fetch_head2head(team_against=team_against)
print(h2h)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# normalize the teams
label_encoder = LabelEncoder()
teams_encoded = label_encoder.fit_transform(opposing_teams)
onehot_encoder = OneHotEncoder(sparse_output=False)
teams_onehot = onehot_encoder.fit_transform(teams_encoded.reshape(-1, 1))

X = teams_onehot
scaler = MinMaxScaler()

# Define weights
weights = []
for game in games:
    matchup = game[headers.index("MATCHUP")]
    home_away = game[headers.index("HOME_AWAY")]
    opponent = matchup.split(' vs. ')[1] if "vs." in matchup else matchup.split(' @ ')[1]

    weight = 1.0  # Default weight for regular games
    if opponent == team_against:
        weight = 1.5 if home_away == "A" else 2.0  # More weight for home games vs. the team
    elif home_away == "H":
        weight = 1.5  # More weight for general home games

    weights.append(weight)

# Apply weights to target variable
cat_array = np.array(cat_array) * np.array(weights)  # Scale performance by weight

y_scaled = scaler.fit_transform(cat_array.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Simpler model
model = Sequential([
    Dense(32, input_dim=X.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')  # Regression output
])

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# Predict and decode
team_encoded = onehot_encoder.transform(label_encoder.transform([team_against]).reshape(-1, 1))
predicted_points = model.predict(team_encoded)
predicted_points = scaler.inverse_transform(predicted_points)  # Decode prediction
print(f"Predicted {player_cat} against {team_against}: {predicted_points[0][0]:.2f}")
