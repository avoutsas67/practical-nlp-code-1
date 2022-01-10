#%%
import requests
import json

#%%
api_key = '------key value here'
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
