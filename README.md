# sleeper-draft-llm-agent

1. create a .env file
2. add a openrouter api key OPENROUTER_API_KEY= (can make one at https://openrouter.ai/docs/api-keys)
3. add the JWT (json web token) that your computer uses for requests to sleeper SLEEPER_AUTH= (can be found from inspect -> network -> filter fetch/XHR -> refresh page -> graphql -> headers -> request headers)
4. edit the config.json to change agent names/models (model names should come from openrouter.ai/models)
5. edit the config.json to enter your draft id from sleeper (sleeper.com/draft/nfl/DRAFT_ID)
