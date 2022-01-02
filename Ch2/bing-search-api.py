#%%
import requests
import json

#%%
api_key = '7d1be776156e45d7b08ca4a9cbe645c0'
example_text = "Hollo, wrld" # the text to be spell-checked
endpoint = "https://api.bing.microsoft.com/v7.0/spellcheck"

data = {'text': example_text}
params = {
    'mkt':'en-us',
    'mode':'proof'
    }
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Ocp-Apim-Subscription-Key': api_key,
    }

#%%
response = requests.post(endpoint, headers=headers, params=params, data=data)

#%%
json_response = response.json()
print(json.dumps(json_response, indent=4))
