# run to process outputs from m1

import json
import sys
import pandas as pd
import requests

input_file = sys.argv[1] # input .json file
output_file_proc = sys.argv[2] # output processed .csv file, possible to go directly to MPNN
batch_id = sys.argv[3]
# output_file = sys.argv[4] # output raw .csv file with score (note you have to replace 10.0 with 0.0 via sed)
# molecule_output_file = sys.argv[5] # output .csv file of just the molecules
# api_token = sys.argv[6]

api_token = 'd684e8a5a5a04497b738da311a3e2a72_9e4ee51ba7014b9f9cf26712b09915fa'
url = "https://app.molecule.one/api/v1/batch-search-result/" + str(batch_id)
headers = {
    "Content-Type": "application/json",
    "Authorization": "ApiToken-v1 " + str(api_token)
}
r = requests.get(url, headers=headers)
print(r)
with open(input_file, 'w') as f: # note, sometimes there's extra information from the raw .json file, may need preprocessing
    json.dump(r.json(), f)

data = json.loads(r.content)

# with open('verification_exploratory.json', 'r', encoding='utf-8') as f: # if .json is saved separately, use this
#     data = json.loads(f.read())

proc_data = []
for dictionary in data:
    if 'error' not in dictionary.values():
        proc_data.append(dictionary)
proc_data = [dict(t) for t in {tuple(d.items()) for d in proc_data}]
for dictionary in proc_data:
    dictionary.pop('status')
    dictionary.pop('reactionCount')
    dictionary.pop('price')
    dictionary.pop('certainty')
    dictionary.pop('startedAt')
    dictionary.pop('finishedAt')
    dictionary.pop('timedOut')
    dictionary.pop('runningTime')


result = pd.DataFrame(proc_data)

# result['result'] = result['result'].astype(float) # raw output
# result.to_csv(output_file, header=False, index=False)

result.replace(10.0, 11.0, inplace=True) # replaces 10.0 for nonsythesiazble molecules as 11, which in turn becomes 0, the one we use now
result['result'] = result['result'].apply(lambda x: abs(x-11))
result.to_csv(output_file_proc, header=False, index=False)

# result.replace(10.0, 0.0, inplace=True) # for binary classification
# result.loc[result['result'] > 0.1, 'result'] = 1.0
# result['result'] = result['result'].astype(int)
# result.to_csv(output_file_binary, header=False, index=False)

# for dictionary in proc_data: # outputs just the molecules
#     dictionary.pop('result')
# result = pd.DataFrame(proc_data)
# result.to_csv(molecule_output_file, header=False, index=False)