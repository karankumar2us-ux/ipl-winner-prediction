import pandas as pd
import numpy as np

print("Loading IPL.csv (this is a large file, may take a moment)...")
ipl = pd.read_csv('/mnt/user-data/uploads/IPL.csv',
                   usecols=['match_id', 'batting_team', 'bowling_team', 'toss_winner',
                            'toss_decision', 'venue', 'match_won_by', 'year', 'team_type'])

print("Loaded:", ipl.shape)

# Keep only normal team matches (exclude international/women's etc if mixed)
ipl = ipl[ipl['team_type'].astype(str).str.contains('club', case=False, na=False) |
           ipl['team_type'].isna()]

# One row per match: get team1, team2, toss info, venue, winner
match_df = ipl.groupby('match_id').first().reset_index()

# Get the two teams playing - take unique batting+bowling teams per match
def get_teams(group):
    teams = pd.unique(pd.concat([group['batting_team'], group['bowling_team']]))
    teams = [t for t in teams if pd.notna(t)][:2]
    while len(teams) < 2:
        teams.append(np.nan)
    return pd.Series({'team1': teams[0], 'team2': teams[1]})

teams_pivot = ipl.groupby('match_id').apply(get_teams).reset_index()

# Merge
final = match_df[['match_id', 'toss_winner', 'toss_decision', 'venue', 'match_won_by', 'year']].merge(
    teams_pivot[['match_id', 'team1', 'team2']], on='match_id'
)

# Drop rows with missing essential info
final = final.dropna(subset=['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'match_won_by'])

# Rename for clarity, match_won_by is the winner
final = final.rename(columns={'match_won_by': 'winner'})

print("\nFinal match-level dataset shape:", final.shape)
print(final.head())
print("\nUnique teams:", final['team1'].nunique())
print("Unique venues:", final['venue'].nunique())
print("\nWinner value counts:\n", final['winner'].value_counts().head(15))

final.to_csv('/home/claude/ipl_project/ipl_matches_real.csv', index=False)
print("\nSaved to ipl_matches_real.csv")
