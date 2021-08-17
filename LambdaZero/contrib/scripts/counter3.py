

import requests
# for i in range(10):
#     print(requests.get("http://34.216.6.165:8000/Counter/decr").json())

# predict image on the remote ray cluster
ray_logo_bytes = requests.get("https://github.com/ray-project/ray/raw/"
                              "master/doc/source/images/ray_header_logo.png").content
resp = requests.post("http://34.216.6.165:8000/image_predict", data=ray_logo_bytes)
print(resp.json())