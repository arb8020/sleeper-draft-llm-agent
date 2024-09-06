import requests
import polars as pl
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
import os
import io
import re
import junkdrawer as jd
import json
from fuzzywuzzy import process

load_dotenv()
sleeper_auth = os.getenv('SLEEPER_AUTH')
key = os.getenv('OPENROUTER_API_KEY')


def create_meta_prompt(pos, csv_data):
    return f"""As an large language model, your task is to design a prompt that you will use to create a fantasy football draft cheatsheet. This cheatsheet will be used alongside real-time data during a live draft with the following parameters:

Scoring: PPR
Number of teams: 12
Starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 1 K, 1 DST, 5 BENCH
Snake draft
Draft position: {pos}

The draft is against other large language models, which means strategies may differ from those used against human players.

Important: You must base all strategies and decisions solely on the provided CSV data and real-time draft information. Discard any prior knowledge or outdated training data that is not directly linked to the 2024 season data or current draft environment.

The prompt you design should instruct the creation of a cheatsheet that can be used flexibly throughout the draft. During each round, you'll have access to:

The cheatsheet you're creating
List of available players with positions, teams, 2024 ADP, 2024 positional ADP, and 2024 projected season total
Players already drafted

Your prompt should guide the creation of a strategy-focused cheatsheet rather than a round-by-round guide or specific player targets. It should incorporate analysis of provided CSV data including analyst opinions, 2024 ADP, 2024 positional ADP, and 2024 projected season totals.

In designing your prompt, consider:

How to create a cheatsheet that complements real-time data effectively
Strategies for balancing positional needs with best player available
Methods for evaluating players' floor vs ceiling potential
Approaches to identifying sleepers and potential reaches
Adapting strategies for different stages of the draft
Adjusting for the specific league format (PPR, 12 teams, given lineup requirements)
Incorporating flexibility based on variable draft position

Additionally, account for the unique nature of drafting against large language models, which may employ non-traditional strategies, highly efficient data processing, or even mistakes like hallucination. Factor this into decisions around reaching for players, targeting positions, and exploiting inefficiencies in LLM-driven drafting behavior.

Your designed prompt should instruct the inclusion of clear usage guidelines within the cheatsheet itself. remember that it

After careful consideration, present your designed prompt enclosed in XML tags using <FantasyFootballCheatsheetPrompt> as the root element. Then, briefly explain your reasoning behind the prompt's structure and content.
<csv_data>
{csv_data}
</csv_data>"""

def create_cheatsheet_prompt(pos, csv_data):
    return f"""You are creating a pre-draft strategy and cheat sheet for yourself to use in an upcoming fantasy football draft. Your goal is to create a flexible, strategic framework that you, as an AI language model, can easily reference and adapt during each round of the draft process. The league has the following specifications:
* Scoring: PPR
* Number of teams: 12
* Starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 1 K, 1 DST, 5 BENCH
* Snake draft
* Draft position: {pos}

During each round of the draft, you will be provided with:
1. This strategy/cheat sheet you're creating now
2. The list of available players, their positions and teams, 2024 ADP, 2024 positional ADP, 2024 projected season total
3. The players you have already drafted

Important: Do not create a round-by-round cheatsheet or specific player targets for each round. Instead, focus on overall strategy and principles that can be applied flexibly throughout the draft.

Using the provided CSV data of analyst opinions, 2024 ADP, 2024 positional ADP, 2024 projected season totals, and your knowledge of fantasy football, create a comprehensive strategy and cheat sheet that emphasizes adaptability and strategic decision-making. Your response should include:

1. An overall draft strategy, considering your draft position and the league settings. Focus on principles like value over replacement, positional scarcity, and balancing "best player available" (BPA) with team needs.

2. Guidelines for assessing and adapting to unexpected draft flows, such as positional runs or highly-ranked players falling unexpectedly.

3. A framework for evaluating value over replacement for each position, considering the league's starting lineup requirements and PPR scoring.

4. Strategies for managing the FLEX position and building bench depth, including how to balance upside and floor when selecting bench players.

5. A method for identifying and prioritizing unique outliers, both potential sleepers (players who might outperform their ADP significantly) and reaches (players whose ADP might be inflated relative to their projected value).

6. Guidelines for balancing your roster as the draft progresses, considering the players you've already drafted and the relative strength of remaining players at each position.

7. Strategies for approaching different stages of the draft (early, middle, late rounds) and how your priorities should shift throughout the draft, without specifying particular players for each round.

8. A framework for quickly assessing the best available player in each round, weighing factors like projected points, positional scarcity, and your team's current composition.

9. Guidelines for when to prioritize upside vs. floor, depending on your draft position and team needs.

10. Any relevant insights or trends from the CSV data that inform your overall strategy, particularly focusing on identifying potential value discrepancies between ADP and projections.

11. A checklist or set of questions to ask yourself each round to ensure you're sticking to your overall strategy while remaining flexible.

12. Specific player outliers:
    a. Identify 5-10 players who are significant positive outliers (potential sleepers or great values) relative to their ADP. Briefly explain why each player is a standout target.
    b. Identify 5-10 players who are significant negative outliers (potential busts or overvalued) relative to their ADP. Briefly explain why each player should be approached with caution or avoided.

Present this information in a clear, organized format that you can easily reference and interpret during each round of the draft. Remember, you're creating this for yourself to use, so structure it in a way that will be most useful for an AI language model during a fast-paced draft scenario where you'll need to make quick, strategic decisions based on the current draft state and available players.

When identifying outlier players, focus on those with the most significant discrepancies between ADP and projected value or those with unique situations that the ADP might not fully capture. Consider factors like changes in team situation, injury recovery, or emerging talents that might be overlooked.

<csv>
{csv_data}
</csv>"""

def generate_draft_prompt(cheat_sheet, drafted_all, drafted_me, pick_num, remaining_data):
    return f"""You are an expert fantasy football analyst participating in a draft. It is now your turn to make a selection. Use the following information to make the best pick for your team:

<cheat_sheet>
    {cheat_sheet}
</cheat_sheet>

<drafted_players>
    <all>
        {drafted_all}
    </all>
    <your_team>
        {drafted_me}
    </your_team>
</drafted_players>

<current_pick>
    {pick_num}
</current_pick>

<remaining_players>
    {remaining_data}
</remaining_players>

<league_settings>
    <scoring>PPR</scoring>
    <teams>12</teams>
    <starting_lineup>
        <QB>1</QB>
        <RB>2</RB>
        <WR>2</WR>
        <TE>1</TE>
        <FLEX>1</FLEX>
        <K>1</K>
        <DST>1</DST>
        <BENCH>5</BENCH>
    </starting_lineup>
    <draft_type>Snake</draft_type>
</league_settings>

Analyze the available players, your team needs, and your draft strategy. 
Consider factors such as positional scarcity, value over replacement, upcoming bye weeks, and potential upside.
Provide your analysis and selection in the following XML format:

<draft_response>
    <thought_process>
        [Detailed analysis of your decision-making process]
    </thought_process>
    <player_selection>
        <name>[Player Name]</name>
        <position>[Position]</position>
        <team>[Team]</team>
    </player_selection>
    <justification>
        [2-3 sentences explaining why this player is the best choice for your team at this point in the draft]
    </justification>
</draft_response>
"""


def draft_player(pid, pick_no, draft_id):

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
    

def send_chat(text, agent, draft_id):
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
      "query": f"mutation create_message($text: String) {{\n        create_message(parent_id: \"1137491335772172288\",client_id: \"0aa65b7e-2840-8aef-1b9b-2b57ac99613f\",parent_type: \"draft\",text: $text) {{\n          attachment\n          author_avatar\n          author_display_name\n          author_real_name\n          author_id\n          author_is_bot\n          author_role_id\n          client_id\n          created\n          message_id\n          parent_id\n          parent_type\n          pinned\n          reactions\n          user_reactions\n          text\n          text_map\n        }}\n      }}",
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
    

def validate_pick(player_id, remaining_df, drafted_pids):
    # Check if player_id exists in remaining_df
    if player_id not in remaining_df['player_id'].values:
        return False, "Player ID not found in available players."
    
    # Check if player has already been drafted
    if player_id in drafted_pids:
        return False, "Player has already been drafted."
    
    return True, "Valid pick."

def parse_draft_decision(decision, data):
    player_name_match = re.search(r'<name>(.*?)</name>', decision)
    if not player_name_match:
        raise ValueError("Could not find player name in the decision")
    
    player_name = player_name_match.group(1)
    
    position_match = re.search(r'<position>(.*?)</position>', decision)
    team_match = re.search(r'<team>(.*?)</team>', decision)
    
    position = position_match.group(1) if position_match else None
    team = team_match.group(1) if team_match else None
    
    player_names = data['fullName'].tolist()
    
    closest_match, match_score = process.extractOne(player_name, player_names)
    
    match_threshold = 80  
    
    if match_score < match_threshold:
        raise ValueError(f"No good match found for player {player_name}. Closest match: {closest_match} with score: {match_score}")
    
    player_row = data[data['fullName'] == closest_match]
    
    if player_row.empty:
        raise ValueError(f"Could not find player {closest_match} in the CSV data")
    
    player_id = player_row.iloc[0]['player_id']
    
    return str(player_id)

def generate_draft_prompt(cheat_sheet, drafted_all, drafted_me, pick_num, remaining_data):
    return f"""You are an expert fantasy football analyst participating in a draft. It is now your turn to make a selection. Use the following information to make the best pick for your team:
Your pre-draft strategy and cheat sheet:
{cheat_sheet}
Available players and their projections:
{remaining_data}
Players that have already been drafted:
{drafted_all}
Players that you drafted:
{drafted_me}
Current pick of the draft: {pick_num}
League settings:
Scoring: PPR
Number of teams: 12
Starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 1 K, 1 DST, 5 BENCH
Snake draft
Analyze the available players, your team needs, and your draft strategy. 
Consider factors such as positional scarcity, value over replacement, upcoming bye weeks, and potential upside.
Provide your analysis and selection in the following XML format:
<draft_response>
<thought_process>
[Detailed analysis of your decision-making process]
</thought_process>
<player_selection>
    <name>[Player Name]</name>
    <position>[Position]</position>
    <team>[Team]</team>
</player_selection>
<justification>
    [2-3 sentences explaining why this player is the best choice for your team at this point in the draft]
</justification>
"""

def openrouter_req(model, prompt, key):
    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": f"Bearer {key}",
      },
      data=json.dumps({
        "model": model, 
        "messages": [
          { "role": "user", "content": prompt}
        ]

      })
    )
    rjs = response.json()
    return rjs

def draft_player_with_validation(agent, model, key, cheat_sheet, drafted_pids, drafted_all, drafted_me, remaining_data, pick_num, draft_id, max_retries=5):
    remaining_df = pd.read_csv(io.StringIO(remaining_data))
    
    initial_prompt = generate_draft_prompt(cheat_sheet, drafted_all, drafted_me, pick_num, remaining_data)
    messages = [{"role": "user", "content": initial_prompt}]
    
    conversation = multi_turn_conversation(model, messages, key, max_retries)
    
    for _ in range(max_retries):
        try:
            decision, conversation_history = next(conversation)
            thoughts = decision.split('<thought_process>')[1].split('</thought_process')[0]
            player_selection = decision.split('<player_selection>')[1].split('</player_selection')[0]
            justification = decision.split('<justification>')[1].split('</justification')[0]
            
            if decision.startswith("Error:"):
                continue  # Skip to next iteration if there was an API error
            message_to_send = thoughts + '\n' + player_selection + '\n' + justification
            send_chat(message_to_send, agent.upper(), draft_id)
            
            try:
                player_id = parse_draft_decision(decision, data=remaining_df)
                is_valid, message = validate_pick(player_id, remaining_df, drafted_pids)
                
                if is_valid:
                    success = draft_player(pid=player_id, pick_no=pick_num, draft_id=draft_id)
                    if success:
                        return player_id, decision, conversation_history
                else:
                    feedback = (f"Your pick of player ID {player_id} ({player_selection}) was invalid: {message}. "
                                f"Please analyze the available players again and make a different selection. "
                                f"Remember to consider your team needs, the current draft position, and the remaining available players.")
                    send_chat('I picked an invalid player. I will try again.', agent.upper(), draft_id)
                    messages.append({"role": "user", "content": feedback})
            except ValueError as e:
                feedback = (f"Error in your pick: {str(e)}. "
                            f"Please make sure you're selecting a valid player from the available players list. "
                            f"Provide your selection in the correct format with the player's full name, position, and team.")
                send_chat('I picked an invalid player. I will try again.', agent.upper(), draft_id)
                messages.append({"role": "user", "content": feedback})
            
        except StopIteration:
            break
    
    # Fallback logic
    fallback_player = remaining_df.sort_values('adp').iloc[0]
    fallback_id = fallback_player['player_id']
    fallback_name = fallback_player['fullName']
    fallback_message = (f"You've made {max_retries} invalid picks. As a fallback, we're selecting {fallback_name}, "
                        f"the highest ADP player available. In future rounds, please be more careful in your player selection.")
    send_chat('Im falling back to choosing the highest ADP player. In future rounds I will be more careful', agent.upper(), draft_id)
    success = draft_player(pid=fallback_id, pick_no=pick_num, draft_id=draft_id)
    if success:
        return fallback_id, fallback_message, conversation_history
    
    return None

def multi_turn_conversation(model, messages, key, max_turns=5):
    conversation_history = []
    
    for _ in range(max_turns):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                },
                json={
                    "model": model,
                    "messages": messages + conversation_history
                }
            )
            
            response.raise_for_status()  
            
            response_data = response.json()
            assistant_message = response_data['choices'][0]['message']['content']
            conversation_history.append({"role": "assistant", "content": assistant_message})
            
            yield assistant_message, conversation_history
            
        except requests.RequestException as e:
            yield f"Error: API request failed - {str(e)}", conversation_history
        
        except Exception as e:
            yield f"Error: Unexpected error - {str(e)}", conversation_history
    
    yield "Error: Max turns reached", conversation_history
