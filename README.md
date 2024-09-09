# sleeper-draft-llm-agent

This repository allows you to conduct a fantasy football draft on Sleeper Fantasy using Large Language Model Agents. The Agents will analyze the draft data CSV in order to come up with prompts for themselves to create cheatsheets/draft strategy guides. They will then use those prompts to create the guides, and then use that guide along with the CSV data in order to draft players against each other in real time. The current setup is for a 12 team PPR draft with 1 QB, 2 RB 2 WR 1 FLEX 1 TE 1 K 1 DST.

## USAGE
1. create a .env file
2. add a openrouter api key OPENROUTER_API_KEY= (can make one at https://openrouter.ai/docs/api-keys)
3. add the JWT (json web token) that your computer uses for requests to sleeper SLEEPER_AUTH= (can be found from inspect -> network -> filter fetch/XHR -> refresh page -> graphql -> headers -> request headers)
4. add the parent_id and client_id that your computer uses for sending messages in the sleeper draft as PARENT_ID and CLIENT_ID (use same process as above)
4. edit the config.json to change agent names/models (model names should come from openrouter.ai/models)
5. edit the config.json to enter your draft id from sleeper (sleeper.com/draft/nfl/DRAFT_ID)
6. run draft_script.py
7. output files will be created

## OVERVIEW
- draft_script.py is the main script that runs the agents
- config.json is where you put which models you want to use, and the draftid (required for above to work)
- create_initial_draft.py is where you would modify the main csv data that the models use to create cheatsheets and draft
- junkdrawer.py has all the random unorganized utility functions
- initialdraftdf.csv is included for easily running the current version

## TODOS:
### cleanup
- test the above process
- make this README more comprehensive/clean
- clean up junkdrawer.py
### features/improvements
- make it more easy to tweak league type/settings
- implement a more robust ACI to reduce agent mistakes (see SWE-Bench paper)
- add more interesting data in create_initial_draft.py
- potentially have the models come up with their own player forecasts
- better prompt engineering
