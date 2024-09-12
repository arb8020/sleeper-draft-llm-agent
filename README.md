# sleeper-draft-llm-agent

This repository is meant to be an exploration of using Large Language Model Agents for tasks like playing Fantasy Football. We use Sleeper Fantasy and ESPN for data. Currently the models are set up for a 12 team PPR League: 1QB, 2RB, 2WR, 1FLEX, 1TE, 1K, 1D

## Key Files
- draft_script_aci.py: implements an ACI inspired by SWE-Agent to allow the models to more easily navigate through data and draft players. can connect to sleeper and draft automatically
- roster_actions_aci.py: similar to the above, allows the models to evaluate their roster, search free agents, and generate add drop decisions. cannot yet connect to sleeper, actions have to be taken manually.
- draft_script.py: first version, had models compress data by making themselves cheatsheets for the draft and drafting using that information

## COMING SOON
- automatic roster actions
- models trading with each other
- setting lineups
- more model forecasting/reasoning

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
- add more interesting data in create_initial_draft.py
- potentially have the models come up with their own player forecasts
- add trade functionality
- add lineup setting
- generally make the LLM do more of the work, very hand-holded at the moment
- better prompt engineering
