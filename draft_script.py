import requests
import polars as pl
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
import os
import io
import re
import json
import junkdrawer as jd
import json
import csv

load_dotenv()
output_dir = f'out/{draft_id}/'
os.makedirs(output_dir, exist_ok=True)


with open('config.json', 'r') as config_file:
    config = json.load(config_file)

agents_models = config["models"]
draft_id = config["draft_id"]
agents = agents_models.keys()

df = pd.read_csv('initialdraftdf.csv')

csv_data = df.to_csv(index=False)
draft_df = pd.read_csv(io.StringIO(csv_data))
csv_data = draft_df.to_csv(index=False)

key = os.getenv('OPENROUTER_API_KEY')

pos = 1
agent_resp = {}
cheatsheet_prompt = {}
print('creating prompts')
for agent in agents_models:
    meta_prompt = jd.create_meta_prompt(pos=pos)
    model = agents_models[agent]
    print(f'agent: {agent} creating meta-prompt ({model})')
    rtime = time.time()
    respjson = jd.openrouter_req(model, meta_prompt, key)
    print(f'agent {agent} ({model}) responded after: {time.time()-rtime}s')
    full_resp = respjson['choices'][0]['message']['content']
    prompt = full_resp.split('<FantasyFootballCheatsheetPrompt>')[1].split('</FantasyFootballCheatsheetPrompt>')[0]
    cheatsheet_prompt[agent] = prompt
    pos += 1
    
print('creating cheatsheets')
pos = 1
for agent in agents_models:
    prompt = cheatsheet_prompt[agent]
    model = agents_models[agent]
    print(f'agent: {agent} creating cheatsheet ({model})')
    rtime = time.time()
    respjson = jd.openrouter_req(model, prompt, key)
    print(f'agent {agent} ({model}) responded after: {time.time()-rtime}s')
    agent_resp[agent] = respjson
    pos += 1
    
with open(os.path.join(output_dir, 'cheatsheet_prompt.json'), 'w') as file:
    json.dump(cheatsheet_prompt, file)

with open(os.path.join(output_dir, 'cheatsheet_map.json'), 'w') as file:
    json.dump(cheatsheet_map, file)
    
cheatsheet_map = {}
for agent in agents_models:
    cheatsheet_map[agent] = agent_resp[agent]['choices'][0]['message']['content']
    
tdf = draft_df[['fullName', 'position', 'player_id', 'adp', 'pos_adp', 'total_proj']]

drafted_pids = []
agent_drafted_pids = {'a': [], 'b': [], 'c': [], 'd': [],
                      'e': [], 'f': [], 'g': [], 'h': [],
                      'i': [], 'j': [], 'k': [], 'l': []}

this_pick_no = 1
this_draft_id = 

snake_agents = agents + agents[::-1]

with open(os.path.join(output_dir, 'draft_responses.txt'), 'a') as response_file:
    while this_pick_no < 169:
        idx = ((this_pick_no - 1) % (len(snake_agents))) if this_pick_no > 24 else this_pick_no-1
        agent = snake_agents[idx]
        cheat_sheet = cheatsheet_map[agent]
        drafted_all = tdf[tdf['player_id'].isin(drafted_pids)].to_csv()
        drafted_me = tdf[tdf['player_id'].isin(agent_drafted_pids[agent])].to_csv()
        remaining_data = tdf[~tdf['player_id'].isin(drafted_pids)].to_csv()
        model = agents_models[agent]
        player_id, decision, conversation_history = jd.draft_player_with_validation(
            agent, model, key, cheat_sheet, drafted_pids, drafted_all, drafted_me, remaining_data, 
            this_pick_no, this_draft_id
        )

        drafted_pids.append(str(player_id))
        agent_drafted_pids[agent].append(str(player_id))
        this_pick_no += 1

        with open('draft_responses.txt', 'a') as response_file:
            response_file.write(f'Agent: {agent}\n')
            response_file.write(f'Pick number: {this_pick_no}\n')
            response_file.write('Conversation history:\n')
            for message in conversation_history:
                response_file.write(f"{message['role'].capitalize()}: {message['content']}\n")
            response_file.write(f'Final decision: {decision}\n')
            response_file.write(f'Drafted player ID: {player_id}\n\n')

        print(f'Agent {agent} completed pick {this_pick_no-1}')
        time.sleep(0.1)
