import json
import junkdrawer as jd
import requests
import numpy as np
import pandas as pd

players_req = requests.get('https://api.sleeper.com/players/nfl/')
psj = players_req.json()

df = pd.DataFrame(psj).T
valid_pos = [['WR'], ['RB'], ['TE'], ['K'], ['QB'], ['QB', 'TE']]
draftable_df = df[(df['active']) & (df['fantasy_positions'].isin(valid_pos))]
ddf = draftable_df
ddf['fullName'] = ddf['first_name'] + ' ' + ddf['last_name']
ddf['player_id'] = ddf['player_id'].str.strip()


offset = 0
espn_url = 'https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/2024/segments/0/leaguedefaults/3?view=kona_player_info'
js_responses = []
while offset < 700:
    print('offset: ', offset)
    espn_filter = jd.create_filter_json(offset)
    espn_headers = {
    'x-fantasy-filter': espn_filter
    }
    print('requesting espn ...')
    espn_req = requests.get(espn_url, headers=espn_headers)
    espnjs = espn_req.json()
    js_responses.append(espnjs)
    offset += 50

player_dicts = [
    pd.DataFrame(js_responses[r]['players'][x])['player'].T[['active', 'id', 'firstName', 'fullName', 'seasonOutlook']].to_dict()
    for r in range(14)
    for x in range(50)
    if 'seasonOutlook' in pd.DataFrame(js_responses[r]['players'][x])['player'].T
]

epdf = pd.DataFrame(player_dicts)

np.setdiff1d(epdf['fullName'].values, ddf['fullName'].values)
name_mapping = {
    'Brian Robinson Jr.': 'Brian Robinson',
    'Brian Thomas Jr.': 'Brian Thomas',
    'Browns D/ST': 'Cleveland Browns',
    'Chiefs D/ST': 'Kansas City Chiefs',
    'Colts D/ST': 'Indianapolis Colts',
    'Cowboys D/ST': 'Dallas Cowboys',
    'DJ Chark Jr.': 'DJ Chark',
    'Deebo Samuel Sr.': 'Deebo Samuel',
    'Dolphins D/ST': 'Miami Dolphins',
    'Jaguars D/ST': 'Jacksonville Jaguars',
    'Jeff Wilson Jr.': 'Jeff Wilson',
    'Jets D/ST': 'New York Jets',
    'Kenneth Walker III': 'Kenneth Walker',
    'Lions D/ST': 'Detroit Lions',
    'Marquise Brown': 'Hollywood Brown',
    'Marvin Harrison Jr.': 'Marvin Harrison',
    'Marvin Mims Jr.': 'Marvin Mims',
    'Michael Penix Jr.': 'Michael Penix',
    'Michael Pittman Jr.': 'Michael Pittman',
    'Odell Beckham Jr.': 'Odell Beckham',
    'Ravens D/ST': 'Baltimore Ravens',
    'Saints D/ST': 'New Orleans Saints',
    'Steelers D/ST': 'Pittsburgh Steelers',
    'Texans D/ST': 'Houston Texans',
    'Travis Etienne Jr.': 'Travis Etienne',
    'Tyrone Tracy Jr.': 'Tyrone Tracy',
    '49ers D/ST': 'San Francisco 49ers',
    'Bears D/ST': 'Chicago Bears', 
    'Bengals D/ST': 'Cincinnati Bengals', 
    'Bills D/ST': 'Buffalo Bills',
}
epdf['fullName_original'] = epdf['fullName']
epdf['fullName'] = epdf['fullName'].replace(name_mapping)

mdf = epdf.merge(ddf, on='fullName', how='left')

# under 1000 API calls per minute
# 1000/60 per second
# wait 
rate_limit_wait_raw = 6/100
rate_limit_wait = rate_limit_wait_raw * 1.01

pid_js_map = {}
for pid in mdf.index.values:
    print(pid)
    rtime = time.time()
    proj_request = requests.get(f'https://api.sleeper.com/projections/nfl/player/{pid}?season_type=regular&season=2024&grouping=week')
    prjs = proj_request.json()
    pid_js_map[pid] = prjs
    used_time = time.time()-rtime
    sleep_time = max(0, rate_limit_wait-used_time)
    time.sleep(sleep_time)
    
pid_data = {}
for pid in pid_js_map.keys():
    this_data = pid_js_map[pid]
    if this_data['1'] is not None:
        adp = this_data['1']['stats']['adp_dd_ppr']
        pos_adp = this_data['1']['stats']['pos_adp_dd_ppr']
        # print(this_data[x])
        total_proj = sum([this_data[x]['stats']['pts_ppr']for x in this_data.keys() if this_data[x] is not None and 'pts_ppr' in this_data[x]['stats'].keys()])
        pid_data[pid] = {'adp': adp, 'pos_adp': pos_adp, 'total_proj': total_proj}
        
draft_data = pd.DataFrame(pid_data).T.reset_index().rename(columns={'index': 'player_id'})
draft_data['player_id'] = draft_data['player_id'].astype(str)

m2df = mdf.merge(draft_data, on='player_id', how='left')

draft_df = m2df[['fullName', 'seasonOutlook', 'position', 'player_id', 'team', 'injury_status', 'adp', 'pos_adp', 'total_proj']]

draft_df.to_csv('initialdraftdf2.csv')
