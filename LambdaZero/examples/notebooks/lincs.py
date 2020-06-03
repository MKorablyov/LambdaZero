import pandas as pd


data = pd.read_json("/home/maksym/Downloads/CD_signatures_binary_42809.json")
#h = data.head()
print(data.columns.values)
# BRD-K66149656
print(data["sig_id"].to_list())