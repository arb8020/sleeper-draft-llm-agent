
import pandas as pd
import json
import os
import time
import re
import requests
from dotenv import load_dotenv
from fuzzywuzzy import process
load_dotenv()

def sleeper_send_chat(text, agent, draft_id, parent_id=os.getenv('PARENT_ID'), client_id=os.getenv('CLIENT_ID'), sleeper_auth=os.getenv('SLEEPER_AUTH')):
    headers = {"authority": 'sleeper.com',
"accept": 'application/json',
"authorization": f'{sleeper_auth}',
"content-type": 'application/json',
"origin": 'https://sleeper.com',
"referer": f'https://sleeper.com/draft/nfl/{draft_id}?ftue=commish',
"user-agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
"x-sleeper-graphql-op": 'create_message',
}
    payload = {
      "operationName": "create_message",
      "query": f"mutation create_message($text: String) {{\n        create_message(parent_id: \"{parent_id}\",client_id: {client_id}\"\",parent_type: \"draft\",text: $text) {{\n          attachment\n          author_avatar\n          author_display_name\n          author_real_name\n          author_id\n          author_is_bot\n          author_role_id\n          client_id\n          created\n          message_id\n          parent_id\n          parent_type\n          pinned\n          reactions\n          user_reactions\n          text\n          text_map\n        }}\n      }}",
      "variables": {
        "text": f"Agent {agent}: {text}"
      }
    }
    
    resp = requests.post('https://sleeper.com/graphql', headers=headers, json=payload)
    resp_js = resp.json()
    if 'errors' in resp_js:
        print(resp_js['errors'])
        return 0
    if resp.status_code == 200:
        return 1
    else:
        return 0
    
def sleeper_draft_player(pid, pick_no, draft_id, sleeper_auth=os.getenv('SLEEPER_AUTH')):

    headers = {"authority": 'sleeper.com',
"accept": 'application/json',
"authorization": f'{sleeper_auth}',
"content-type": 'application/json',
"origin": 'https://sleeper.com',
"referer": f'https://sleeper.com/draft/nfl/{draft_id}?ftue=commish',
"user-agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
"x-sleeper-graphql-op": 'draft_pick_player',
}
    payload = {
        "operationName": "draft_pick_player",
        "query": f"""
        mutation draft_pick_player {{
            draft_pick_player(sport: "nfl", player_id: "{pid}", draft_id: "{draft_id}", pick_no: {pick_no}) {{
                draft_id
                pick_no
                player_id
                picked_by
                is_keeper
                metadata
                reactions
            }}
        }}
        """,
        "variables": {}
    }
    resp = requests.post('https://sleeper.com/graphql', headers=headers, json=payload)
    resp_js = resp.json()
    if 'errors' in resp_js:
        print(resp_js['errors'])
        return 0
    if resp.status_code == 200:
        return 1
    else:
        return 0
    

def debug_print(message, level, current_debug_level):
    if current_debug_level >= level:
        print(message)
        
def openrouter_req(model, messages, key):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
        },
        json={
            "model": model,
            "messages": messages
        }
    )
    return response.json()

def manage_conversation_history(history, new_message, max_messages=10):
    if len(history) >= max_messages:
        # Summarize older messages
        summary = {"role": "system", "content": f"[{len(history) - max_messages + 1} earlier messages omitted]"}
        history = [summary] + history[-(max_messages-1):]
    
    history.append(new_message)
    return history

def get_top_players(df, drafted_pids, n=10, original_metric='adp', position=None):
    original_metric = original_metric.strip("'").strip('"')
    df = df[~df['player_id'].isin(drafted_pids)]
    
    if type(n) != int:
        n = int(n)
        
    if position is not None:
        best_match_pos = process.extractOne(position, choices=['RB', 'WR', 'QB', 'TE', 'K', 'DEF'])
        pos = best_match_pos[0]
        df = df[df['position']==pos]
    
    best_match_metric = process.extractOne(original_metric, choices=["adp", "pos_adp", "total_proj"])
    metric = best_match_metric[0]
    
    if metric == 'total_proj':
        return df.sort_values(metric, ascending=False).head(int(n))[['player_id', 'fullName', 'position', metric]].to_dict('records')
    else:
        return df.sort_values(metric).head(int(n))[['player_id', 'fullName', 'position', metric]].to_dict('records')

def get_player_info(df, player_id):
    return df[df['player_id'] == player_id].to_dict('records')[0]

def get_team_roster(df, drafted_pids):
    return df[df['player_id'].isin(drafted_pids)][['player_id', 'fullName', 'position']].to_dict('records')

def get_draft_status(pick_number):
    round_number = (pick_number - 1) // 12 + 1
    pick_in_round = (pick_number - 1) % 12 + 1
    return {"round": round_number, "pick_in_round": pick_in_round, "overall_pick": pick_number}

def parse_command(command_string):
    # - A command name (letters, numbers, underscores)
    # - An opening parenthesis
    # - Optional parameters which can be numbers or quoted strings
    # - A closing parenthesis
    pattern = r'(\w+)\(([^)]*)\)'
    
    match = re.match(pattern, command_string)
    if match:
        command = match.group(1)
        params = re.findall(r'\s*(?:(\'[^\']*\')|([^\s,]+))\s*', match.group(2))
        params = [(param[0].strip("'") if param[0] else int(param[1]) if param[1].isdigit() else param[1].strip()) for param in params]
        return command, params
    else:
        return None, None
    
def parse_lm_response(response):
    parts = response.split('Action:', 1)
    thought = parts[0].replace('Thought:', '').strip()
    action = parts[1].strip() if len(parts) > 1 else ''
    return thought, action

def execute_command(command, params, df, drafted_pids, agent_rosters, agent, pick_number):
    if command == "get_top_players":
        return get_top_players(df, drafted_pids,  *params)
    elif command == "get_player_info":
        return get_player_info(df, *params)
    elif command == "get_team_roster":
        return get_team_roster(df, agent_rosters[agent])
    elif command == "get_draft_status":
        return get_draft_status(pick_number)
    else:
        return f"Unknown command: {command}"

def get_current_context(agent, df, drafted_pids, pick_number):
    context = {
        "draft_status": get_draft_status(pick_number),
        "drafted_players": drafted_pids,
        "team_roster": get_team_roster(df, agent_rosters[agent]),
        "available_positions": df[~df['player_id'].isin(drafted_pids)]['position'].value_counts().to_dict()
    }
    return json.dumps(context)

def manage_history(history, new_observation, max_observations=5):
    """
    Manage the history of observations.
    Keep the most recent observations and collapse older ones.
    """
    if len(history) >= max_observations:
        collapsed = f"[{len(history) - max_observations + 1} earlier observations]"
        history = [collapsed] + history[-(max_observations-1):]
    
    history.append(new_observation)
    return history

def draft_player_aci(agent, model, key, df, drafted_pids, pick_number, debug_level=0):
    context = get_current_context(agent, df, drafted_pids, pick_number)
    
    debug_print(f"Debug Level: {debug_level}", 1, debug_level)
    debug_print(f"Current Context: {context}", 2, debug_level)
    
    system_message = {
        "role": "system",
        "content": """You are an AI assistant helping with a fantasy football draft. 
        You can use the following commands to get information:
        - get_top_players(n, metric, position=None): Returns top n available players sorted by a given metric (adp, pos_adp, total_proj), with an option to filter by position.
        - ex1: get_top_players(5, 'adp')
        - ex1: get_top_players(5, 'total_proj', position='TE')
        - ex1: get_top_players(5, 'pos_adp', position='WR')
        - get_player_info(player_id): Returns detailed info for a specific player
        - get_team_roster(): Returns the current roster of your team
        - get_draft_status(): Returns current draft round, pick, etc.
        - draft_player(player_id): Attempts to draft player

        Respond with your thought process and then an action using one of these commands.
        Format your response as:
        Thought: [Your reasoning here]
        Action: [command_name(parameters)]
        """
    }
    
    conversation_history = [system_message]
    
    user_message = {
        "role": "user",
        "content": f"It's your turn to draft. Here's the current context:\n{context}\n\nWhat would you like to do?"
    }
    conversation_history = manage_conversation_history(conversation_history, user_message)

    debug_print("Initial conversation history created", 2, debug_level)

    while True:
        debug_print("Sending request to OpenRouter API", 1, debug_level)
        response = openrouter_req(model, conversation_history, key)
        assistant_message = response['choices'][0]['message']
        thought, action = parse_lm_response(assistant_message['content'])
        
        debug_print(f"Thought: {thought}", 1, debug_level)
        debug_print(f"Action: {action}", 1, debug_level)
        
        debug_print(f"Full assistant message: {assistant_message}", 2, debug_level)
        
        conversation_history = manage_conversation_history(conversation_history, assistant_message)
        
        if not action:
            debug_print("No action provided, skipping pick", 1, debug_level)
            return None

        command, params = parse_command(action)
        
        debug_print(f"Executing command: {command} with params: {params}", 1, debug_level)
        result = execute_command(command, params, df, drafted_pids, agent_rosters, agent, pick_number)
        debug_print(f"Result: {result}", 1, debug_level)
        
        system_message = {
            "role": "system",
            "content": f"Command result: {result}"
        }
        conversation_history = manage_conversation_history(conversation_history, system_message)
        
        debug_print("Updated conversation history", 2, debug_level)
        
        user_message = {
            "role": "user",
            "content": "Based on this information, would you like to draft a player or get more information? If drafting, specify the player_id."
        }
        conversation_history = manage_conversation_history(conversation_history, user_message)
        
        debug_print("Sending follow-up request to OpenRouter API", 1, debug_level)
        response = openrouter_req(model, conversation_history, key)
        assistant_message = response['choices'][0]['message']
        thought, action = parse_lm_response(assistant_message['content'])
        
        debug_print(f"Follow-up response - Thought: {thought}, Action: {action}", 1, debug_level)
        
        conversation_history = manage_conversation_history(conversation_history, assistant_message)
        
        if 'draft_player' in action:
            player_id = action.split('(')[1].rstrip(')').strip().strip("'").strip('"')
            # print('pid', player_id)
            # print('valid', player_id in df['player_id'].values)
            # print('not drafted', player_id not in drafted_pids)
            if player_id in df['player_id'].values and player_id not in drafted_pids:
                debug_print(f"Valid draft attempt for player_id: {player_id}", 1, debug_level)
                return player_id, conversation_history
            else:
                debug_print(f"Invalid draft attempt for player_id: {player_id}", 1, debug_level)
                system_message = {
                    "role": "system",
                    "content": f"Invalid player_id: {player_id}"
                }
                conversation_history = manage_conversation_history(conversation_history, system_message)
        
        user_message = {
            "role": "user",
            "content": "You decided to get more information. What would you like to do next?"
        }
        conversation_history = manage_conversation_history(conversation_history, user_message)
        debug_print("Continuing to next iteration", 1, debug_level)
        
# Main script
debug_level = 0
df = pd.read_csv('initialdraftdf.csv')
key = os.getenv('OPENROUTER_API_KEY')
sleeper_auth = os.getenv('SLEEPER_AUTH')


draft_id = int(time.time()) # change this

agents_models = config["models"]
draft_id = config["draft_id"]

output_dir = f'out/{draft_id}/'
os.makedirs(output_dir, exist_ok=True)

drafted_pids = []
agent_rosters = {agent: [] for agent in agents_models.keys()}
agents = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
snake_agents = agents + agents[::-1]

for pick_number in range(1, 169):
    idx = ((pick_number - 1) % (len(agents_models))) if pick_number > 24 else pick_number-1
    agent = list(snake_agents)[idx]
    model = agents_models[agent]
    
    print(f"\nPick {pick_number}: Agent {agent}'s turn")
    player_id, conversation_history = draft_player_aci(agent, model, key, df, drafted_pids, pick_number, debug_level = debug_level)
    
    if player_id:
        agent_rosters[agent].append(player_id)
        drafted_pids.append(player_id)
        player_name = df[df['player_id'] == player_id]['fullName'].values[0]
        agent_thoughts = 'AGENT ' + str(agent).upper() + ': ' + '\n\n'.join([x['content'] for x in conversation_history if x['role']=='assistant'])
        debug_print(agent_thoughts, 0, debug_level)
        debug_print(f"Agent {agent} drafted player {player_name} (ID: {player_id})\n", 0, debug_level)
        # sleeper_draft_player(player_id, pick_number, draft_id)
        # sleeper_send_chat(agent_thoughts, agent, draft_id)
    else:
        print(f"Agent {agent} skipped their pick")
    
    with open(os.path.join(output_dir, 'draft_results.json'), 'w') as f:
        json.dump(drafted_pids, f)

print("\nFinal Draft Results:")
for agent, roster in zip(agents_models.keys(), [drafted_pids[i::12] for i in range(12)]):
    print(f"\nAgent {agent}'s Team:")
    for pid in roster:
        player = df[df['player_id'] == pid].iloc[0]
        print(f"{player['fullName']} ({player['position']})")
