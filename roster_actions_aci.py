import pandas as pd
import requests
import os
import re
import json
from dotenv import load_dotenv
from fuzzywuzzy import process, fuzz
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
sleeper_auth = os.getenv('SLEEPER_AUTH')
league_id = os.getenv('LEAGUE_ID')
key = os.getenv('OPENROUTER_API_KEY')

# Constants
HEADERS = {
    "authority": 'sleeper.com',
    "accept": 'application/json',
    "authorization": f'{sleeper_auth}',
    "content-type": 'application/json',
    "origin": 'https://sleeper.com',
    "referer": f'https://sleeper.com/leagues/{league_id}/matchup',
    "user-agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
    "x-sleeper-graphql-op": 'draft_pick_player',
}
agents = [x for x in 'abcdefghijkl']
AGENTS_MODELS = {agent: "openai/gpt-4o-mini" for agent in 'abcdefghijkl'}

# Helper functions
def openrouter_req(model, messages, key, max_retries=3, retry_delay=1):
    """
    Make a request to OpenRouter API with retry mechanism.
    
    :param model: The model to use for the request
    :param messages: The messages to send to the model
    :param key: The API key
    :param max_retries: Maximum number of retry attempts (default 3)
    :param retry_delay: Delay between retries in seconds (default 1)
    :return: JSON response from the API
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages},
                timeout=10  # Add a timeout to prevent indefinite hanging
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except (requests.RequestException, ValueError) as e:
            if attempt < max_retries - 1:
                print(f"Request failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Last error: {str(e)}")
                return {"error": str(e)}  

def manage_conversation_history(history, new_message, max_messages=10):
    if len(history) >= max_messages:
        summary = {"role": "system", "content": f"[{len(history) - max_messages + 1} earlier messages omitted]"}
        history = [summary] + history[-(max_messages-1):]
    history.append(new_message)
    return history

def parse_lm_response(response):
    parts = response.split('Action:', 1)
    thought = parts[0].replace('Thought:', '').strip()
    action = parts[1].strip() if len(parts) > 1 else ''
    return thought, action

def parse_command(command_string):
    pattern = r'(\w+)\(([^)]*)\)'
    match = re.match(pattern, command_string)
    if match:
        command = match.group(1)
        params = re.findall(r'\s*(?:(\'[^\']*\')|([^\s,]+))\s*', match.group(2))
        params = [(param[0].strip("'") if param[0] else int(param[1]) if param[1].isdigit() else param[1].strip()) for param in params]
        return command, params
    else:
        return None, None

def get_league_detail(league_id, headers):
    payload = {
        "operationName": "get_league_detail",
        "variables": {},
        "query": """
        query get_league_detail {
            league_rosters(league_id: "%s") {
                league_id
                metadata
                owner_id
                co_owners
                players
                roster_id
                settings
                starters
                keepers
                reserve
                taxi
                player_map
            }
            league_users(league_id: "%s") {
                avatar
                user_id
                league_id
                metadata
                settings
                display_name
                is_owner
                is_bot
            }
            league_transactions_filtered(
                league_id: "%s",
                roster_id_filters: [],
                type_filters: [],
                leg_filters: [],
                status_filters: ["pending", "proposed"]
            ) {
                adds
                consenter_ids
                created
                creator
                draft_picks
                drops
                league_id
                leg
                metadata
                roster_ids
                settings
                status
                status_updated
                transaction_id
                type
                player_map
                waiver_budget
            }
            matchup_legs_1: matchup_legs(
                league_id: "%s",
                round: 1
            ) {
                league_id
                leg
                matchup_id
                roster_id
                round
                starters
                players
                player_map
                points
                proj_points
                max_points
                custom_points
                starters_games
                picks
                bans
                subs
            }
        }
        """ % (league_id, league_id, league_id, league_id)
    }

    resp = requests.post('https://sleeper.com/graphql', headers=headers, json=payload)
    return resp.json()

def get_top_players_stats(fa_pos_stats, n, metric, position):
    print('running stats')
    position_players = fa_pos_stats[position]
    top_players = position_players.sort_values(metric, ascending=False).head(int(n))
    print(f'found {len(top_players)} in {position} by {metric}')
    return top_players[['full_name', 'player_id', metric]].to_dict('records')

def get_top_players_projections(fa_pos_proj, n, metric, position):
    print('running projections')
    position_players = fa_pos_proj[position]
    top_players = position_players.sort_values(metric, ascending=False).head(int(n))
    print(f'found {len(top_players)} in {position} by {metric}')
    return top_players[['full_name', 'player_id', metric]].to_dict('records')

def get_player_info(player_id, projs, stats):
    player_id = str(player_id)
    pos_proj_cols = {}

    pos_proj_cols['DEF'] = ['pts_ppr', 'safe', 'int', 'pts_allow', 'yds_allow', 'tkl_loss']
    pos_proj_cols['QB'] = ['pts_ppr', 'pass_yd', 'cmp_pct', 'pass_td', 'rush_att', 'rush_td', 'fum']
    pos_proj_cols['K'] = ['pts_ppr','fga', 'fgm', 'fgm_20_29', 'fgm_40_49', 'fgm_50p']
    pos_proj_cols['TE'] = ['pts_ppr', 'rec_tgt', 'rec', 'rec_yd', 'rec_td',]
    pos_proj_cols['RB'] = ['pts_ppr', 'rush_att', 'rush_yd',  'rush_td', 'rec_tgt', 'rec', 'rec_yd', ]
    pos_proj_cols['WR'] = ['pts_ppr', 'rec_tgt', 'rec', 'rec_yd', 'rec_td', 'rush_att', 'rush_yd',]

    pos_stats_cols = {}
    
    pos_stats_cols['DEF'] = ['pts_ppr', 'qb_hit', 'def_forced_punts', 'def_st_ff', 'safe', 'int', 'pts_allow', 'yds_allow', 'tkl_loss', 'def_st_td']
    pos_stats_cols['QB'] = ['pts_ppr', 'pass_yd', 'cmp_pct', 'pass_ypc', 'pass_rz_att', 'pass_td', 'rush_att', 'rush_rec_yd', 'rush_rz_att', 'rush_td', 'fum']
    pos_stats_cols['K'] = ['pts_ppr','fga', 'fgm', 'fgm_pct', 'fgm_20_29', 'fgm_40_49', 'fgm_50p']
    pos_stats_cols['TE'] = ['pts_ppr', 'rec_tgt', 'rec', 'rec_yd', 'rec_rz_tgt', 'rec_td', 'off_snp', 'tm_off_snp']
    pos_stats_cols['RB'] = ['pts_ppr', 'rush_att', 'rush_yd', 'rush_rz_att', 'rush_td', 'rec_tgt', 'rec', 'rec_yd', 'rec_rz_tgt', 'off_snp', 'tm_off_snp']
    pos_stats_cols['WR'] = ['pts_ppr', 'rec_tgt', 'rec', 'rec_yd', 'rec_rz_tgt', 'rec_td', 'rush_att', 'rush_yd', 'rush_rz_att', 'off_snp', 'tm_off_snp']


    if player_id in stats['player_id'].unique() and player_id in projs['player_id'].unique():
        player_data_stats = stats[stats['player_id'] == player_id].iloc[0]['stats']
        position = stats[stats['player_id'] == player_id]['player'].values[0]['fantasy_positions'][0]
        player_data_projs = projs[projs['player_id'] == player_id].iloc[0]['stats']
        relevant_stats = {stat: player_data_stats[stat] for stat in pos_stats_cols[position] if stat in player_data_stats}
        relevant_proj = {stat: player_data_projs[stat] for stat in pos_proj_cols[position] if stat in player_data_projs}
        return {
        'player_id': player_id,
        'full_name': stats[stats['player_id'] == player_id]['full_name'].values[0],
        'position': position,
        'stats': relevant_stats,
        'proj': relevant_proj
        }
    elif player_id in stats['player_id'].unique():
        player_data_stats = stats[stats['player_id'] == player_id].iloc[0]['stats']
        position = stats[stats['player_id'] == player_id]['player'].values[0]['fantasy_positions'][0]
        relevant_stats = {stat: player_data_stats[stat] for stat in pos_stats_cols[position] if stat in player_data_stats}
        relevant_proj = {stat: 0 for stat in pos_proj_cols[position]}
        return {
        'player_id': player_id,
        'full_name': stats[stats['player_id'] == player_id]['full_name'].values[0],
        'position': position,
        'stats': relevant_stats,
        'proj': relevant_proj
        }
    elif player_id in projs['player_id'].unique():
        player_data_projs = projs[projs['player_id'] == player_id].iloc[0]['stats']
        position = projs[projs['player_id'] == player_id]['player'].values[0]['fantasy_positions'][0]
        relevant_stats = {stat: 0 for stat in pos_stats_cols[position]}
        relevant_proj = {stat: player_data_projs[stat] for stat in pos_proj_cols[position] if stat in player_data_projs}
        return {
        'player_id': player_id,
        'full_name': projs[projs['player_id'] == player_id]['full_name'].values[0],
        'position': position,
        'stats': relevant_stats,
        'proj': relevant_proj
        }
    else:
        return -1
    
    
def get_roster_info(roster_ids, projs, stats):
    return [get_player_info(pid, projs, stats) for pid in roster_ids]


def get_rosters(league_detail):
    rosters = {}
    rosters_rid = {}
    for i in range(12):
        this_data = league_detail['data']['league_rosters'][i]
        oid = this_data['owner_id']
        rid = this_data['roster_id']
        players = this_data['player_map'].keys()
        
        rosters[oid] = players
        rosters_rid[rid] = players
    return rosters, rosters_rid

def get_free_agents(all_rostered_ids, stats, projs, active_ids):
    active_stats = stats[stats['player_id'].isin(active_ids)]
    fa_stats = active_stats[~active_stats['player_id'].isin(all_rostered_ids)]
    fa_stats['fantasy_position'] = fa_stats['player'].apply(lambda x: x['fantasy_positions'][0] if 'fantasy_positions' in x else None)
    fa_stats['first_name'] = fa_stats['player'].apply(lambda x: x['first_name'] if 'first_name' in x else None)
    fa_stats['last_name'] = fa_stats['player'].apply(lambda x: x['last_name'] if 'last_name' in x else None)
    fa_stats['full_name'] = fa_stats['first_name'] + ' ' + fa_stats['last_name']

    active_proj = projs[projs['player_id'].isin(active_ids)]
    fa_proj = active_proj[~active_proj['player_id'].isin(all_rostered_ids)]
    fa_proj['fantasy_position'] = fa_proj['player'].apply(lambda x: x['fantasy_positions'][0] if 'fantasy_positions' in x else None)
    fa_proj['first_name'] = fa_proj['player'].apply(lambda x: x['first_name'] if 'first_name' in x else None)
    fa_proj['last_name'] = fa_proj['player'].apply(lambda x: x['last_name'] if 'last_name' in x else None)
    fa_proj['full_name'] = fa_proj['first_name'] + ' ' + fa_proj['last_name']

    return fa_stats, fa_proj

def get_position_data(fa_stats, fa_proj):
    fa_pos_proj = {}
    fa_pos_stats = {}
    for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
        fa_pos_proj[pos] = fa_proj[fa_proj['fantasy_position']==pos]
        fa_pos_proj[pos] = pd.concat([fa_pos_proj[pos].drop('stats', axis=1).reset_index(drop=True), pd.json_normalize(fa_pos_proj[pos]['stats'])], axis=1)
        
        fa_pos_stats[pos] = fa_stats[fa_stats['fantasy_position']==pos]
        fa_pos_stats[pos] = pd.concat([fa_pos_stats[pos].drop('stats', axis=1).reset_index(drop=True), pd.json_normalize(fa_pos_stats[pos]['stats'])], axis=1)    
    
    return fa_pos_proj, fa_pos_stats

def evaluate_roster(roster, projs, stats):
    """
    Evaluate the current roster.
    """
    roster_info = get_roster_info(roster, projs=projs, stats=stats)
    
    system_message = {
        "role": "system",
        "content": f"""You are an AI assistant evaluating a fantasy football roster. 
        Analyze the following roster and provide a brief evaluation:
        
        {json.dumps(roster_info, indent=2)}
        
        Consider the scoring system (PPR) and lineup requirements 
        (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 1 K, 1 DST, 5 BENCH).
        Provide your evaluation in JSON format with the following structure:
        {{
            "team_strength": "score from 1-10",
            "key_players": [
                {{"name": "Player Name", "position": "POS", "role": "description"}},
                ...
            ],
            "weaknesses": ["position1", "position2", ...],
            "overall_assessment": "brief description"
        }}
        """
    }
    
    messages = [system_message]
    response = openrouter_req(AGENTS_MODELS['a'], messages, key)
    evaluation = response['choices'][0]['message']['content']
    
    return evaluation

def source_free_agents(agent, model, key, position, fa_pos_stats, fa_pos_proj):
    """
    Source free agents for a given position.
    """
    metrics = {
        'QB': ['pts_ppr', 'pass_yd', 'pass_td', 'rush_yd'],
        'RB': ['pts_ppr', 'rush_yd', 'rush_td', 'rec_yd'],
        'WR': ['pts_ppr', 'rec_yd', 'rec_td', 'rec'],
        'TE': ['pts_ppr', 'rec_yd', 'rec_td', 'rec'],
        'K': ['pts_ppr', 'fgm', 'fgm_40_49', 'fgm_50p'],
        'DEF': ['pts_ppr', 'sack', 'int',]
    }
    
    sourced_players = set()
    for metric in metrics[position]:
        top_stats = get_top_players_stats(fa_pos_stats, 3, metric, position)
        top_proj = get_top_players_projections(fa_pos_proj, 3, metric, position)
        sourced_players.update([player['player_id'] for player in top_stats + top_proj])
    
    return list(sourced_players)

def evaluate_free_agents(agent, model, key, free_agents, position, projs, stats):
    """
    Evaluate sourced free agents for a given position.
    """
    fa_info = [get_player_info(pid, projs, stats) for pid in free_agents]
    
    system_message = {
        "role": "system",
        "content": f"""You are an AI assistant evaluating free agent {position}s in fantasy football. 
        Analyze the following free agents and provide a brief evaluation:
        
        {json.dumps(fa_info, indent=2)}
        
        Provide your evaluation in JSON format with the following structure:
        {{
            "top_targets": [
                {{"name": "Player Name", "reason": "Why they should be targeted"}},
                ...
            ],
            "sleepers": [
                {{"name": "Player Name", "potential": "Why they have upside"}},
                ...
            ],
            "overall_quality": "Rating of the free agent pool's quality (1-10 scale)"
        }}
        """
    }
    
    messages = [system_message]
    response = openrouter_req(model, messages, key)
    evaluation = response['choices'][0]['message']['content']
    
    return evaluation

def optimize_roster(agent, model, key, roster, roster_evaluation, free_agent_evaluation, total_faab=100, remaining_faab=100):
    """
    Optimize the roster based on current roster evaluation and free agent evaluation.
    """
    system_message = {
        "role": "system",
        "content": f"""You are an AI assistant optimizing a fantasy football roster. 
        Analyze the current roster and available free agents, then propose a single add/drop transaction and FAAB bid to improve the team.
        
        Current roster evaluation:
        {json.dumps(roster_evaluation, indent=2)}
        
        Free agent evaluation:
        {json.dumps(free_agent_evaluation, indent=2)}
        
        Total FAAB: {total_faab}
        Remaining FAAB: {remaining_faab}
        
        Provide your recommendation in JSON format with the following structure:
        {{
            "add_player": "Player Name",
            "drop_player": "Player Name",
            "faab_bid": 0,
            "reasoning": "Explanation for the proposed transaction"
        }}
        """
    }
    
    messages = [system_message]
    response = openrouter_req(model, messages, key)
    recommendation = response['choices'][0]['message']['content']
    
    return recommendation

def execute_transaction(add_player, drop_player, faab_bid): # does not work lol. need to figure out auth
    """
    Execute the proposed transaction.
    """
    add_id = name_id_map.get(add_player)
    drop_id = name_id_map.get(drop_player)
    
    if not add_id or not drop_id:
        return {"status": "error", "message": "Player not found"}
    
    payload = {
        "operationName": "league_create_transaction",
        "query": f"""mutation league_create_transaction($k_adds: [String], $v_adds: [Int], $k_drops: [String], $v_drops: [Int]) {{
            league_create_transaction(
                league_id: "{league_id}",
                type: "free_agent",
                k_adds: $k_adds,
                v_adds: $v_adds,
                k_drops: $k_drops,
                v_drops: $v_drops
            ){{
                adds
                drops
                transaction_id
            }}
        }}""",
        "variables": {
            "k_adds": [add_id],
            "v_adds": [faab_bid],
            "k_drops": [drop_id],
            "v_drops": [3]
        }
    }

    resp = requests.post('https://sleeper.com/graphql', headers=HEADERS, json=payload)
    return resp.json()

def main():
    league_detail = get_league_detail(league_id, HEADERS)
    rosters, rosters_rid = get_rosters(league_detail)

    agent_rosters = {agents[i]: rosters_rid[i+1] for i in range(len(agents))}

    players_req = requests.get('https://api.sleeper.com/players/nfl/')
    psj = players_req.json()

    df_raw = pd.DataFrame(psj).T
    valid_pos = [['WR'], ['RB'], ['TE'], ['K'], ['DEF'], ['QB'], ['QB', 'TE']]
    df = df_raw[(df_raw['active']) & (df_raw['fantasy_positions'].isin(valid_pos))]
    df['full_name'] = df['first_name'] + ' ' + df['last_name']
    active_ids = df['player_id'].values

    proj_url = 'https://api.sleeper.com/projections/nfl/2024/2?season_type=regular&position[]=DEF&position[]=K&position[]=QB&position[]=RB&position[]=TE&position[]=WR&order_by=pts_ppr'
    proj_resp = requests.get(proj_url)
    prjs = proj_resp.json()
    projs = pd.DataFrame(prjs)
    projs['fantasy_position'] = projs['player'].apply(lambda x: x['fantasy_positions'][0] if 'fantasy_positions' in x else None)
    projs['first_name'] = projs['player'].apply(lambda x: x['first_name'] if 'first_name' in x else None)
    projs['last_name'] = projs['player'].apply(lambda x: x['last_name'] if 'last_name' in x else None)
    projs['full_name'] = projs['first_name'] + ' ' + projs['last_name']

    stats_url = 'https://api.sleeper.com/stats/nfl/2024/1?season_type=regular&position[]=DEF&position[]=K&position[]=QB&position[]=RB&position[]=TE&position[]=WR&order_by=pts_ppr'
    stats_resp = requests.get(stats_url)
    srjs = stats_resp.json()
    stats = pd.DataFrame(srjs)
    stats['fantasy_position'] = stats['player'].apply(lambda x: x['fantasy_positions'][0] if 'fantasy_positions' in x else None)
    stats['first_name'] = stats['player'].apply(lambda x: x['first_name'] if 'first_name' in x else None)
    stats['last_name'] = stats['player'].apply(lambda x: x['last_name'] if 'last_name' in x else None)
    stats['full_name'] = stats['first_name'] + ' ' + stats['last_name']

    all_rostered_ids = [key.strip('"').strip("'") for roster in agent_rosters for key in agent_rosters[roster]]

    fa_stats, fa_proj = get_free_agents(all_rostered_ids, stats, projs, active_ids)
    fa_pos_proj, fa_pos_stats = get_position_data(fa_stats, fa_proj)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for agent, model in AGENTS_MODELS.items():
        print(f'analyzing agent: {agent}')
        
        roster_evaluation = evaluate_roster(agent_rosters[agent], projs=projs, stats=stats)
        
        with open(f'roster_evaluation_{agent}_{timestamp}.txt', 'w') as f:
            f.write(roster_evaluation)
        
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            print(f'analyzing {position}s...')
            
            sourced_free_agents = source_free_agents(agent, model, key, position, fa_pos_stats, fa_pos_proj)
            free_agent_evaluation = evaluate_free_agents(agent, model, key, sourced_free_agents, position, projs, stats)
            
            with open(f'free_agent_evaluation_{agent}_{position}_{timestamp}.txt', 'w') as f:
                f.write(free_agent_evaluation)
            
            recommendation = optimize_roster(agent, model, key, agent_rosters[agent], roster_evaluation, free_agent_evaluation)
            
            with open(f'recommendation_{agent}_{position}_{timestamp}.txt', 'w') as f:
                f.write(json.dumps(recommendation, indent=2))
            
            print(f"recommendation for {agent} - {position}:")
            print(json.dumps(recommendation, indent=2))
            
            # result = execute_transaction(recommendation['add_player'], recommendation['drop_player'], recommendation['faab_bid'])
            # print(f"transaction result: {result}")

    print("roster analysis and optimization complete.")

if __name__ == "__main__":
    main()