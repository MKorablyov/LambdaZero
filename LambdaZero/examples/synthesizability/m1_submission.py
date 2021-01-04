# run to submit to m1

import sys
import requests
import json

input = sys.argv[1] # input .csv file
response = sys.argv[2] # saves responses from submission
# processed_input = sys.argv[3] # output .json file that could also be submitted to molecule.one separately, if needed
# api_token = sys.argv[4] # api token to used for submission
api_token = 'd684e8a5a5a04497b738da311a3e2a72_9e4ee51ba7014b9f9cf26712b09915fa'

f = open(input, 'r', encoding='utf-8-sig')
temp = f.read().splitlines()
molecules = []
for line in temp:
    molecules.append(line.split(',')[0])

# payload = {'targets': molecules} # in case the payload is saved separately
# with open(processed_input, 'w') as f:
#     json.dump(payload, f)

url = "https://app.molecule.one/api/v1/batch-search"
headers = {
    "Content-Type": "application/json",
    "Authorization": "ApiToken-v1 " + api_token
}
params = {"exploratory_search": True}
r = requests.post(url, json={"targets": molecules, "params": params}, headers=headers)

with open(response, 'w') as f:
    json.dump(r.json(), f)