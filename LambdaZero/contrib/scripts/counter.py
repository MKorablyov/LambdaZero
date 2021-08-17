import time, socket

import ray
from fastapi import FastAPI
from ray import serve
import requests
app = FastAPI()

@serve.deployment
@serve.ingress(app)
class Counter:
  def __init__(self):
      self.count = 0

  @app.get("/")
  def get(self):
      return {"count": self.count}

  @app.get("/incr")
  def incr(self):
      self.count += 1
      return {"count": self.count}

  @app.get("/decr")
  def decr(self):
      self.count -= 1
      return {"count": self.count}

ray.init(address="auto", namespace="serve")
serve.start(http_options={"host": "0.0.0.0", "port":8000}, detached=True)
Counter.deploy()
for i in range(10):
    print(requests.get("http://127.0.0.1:8000/Counter/incr").json()) # query host


# for i in range(10):
#     print(requests.get("http://172.31.28.208:8000/Counter/incr").json()) # query private ip
#     print(requests.get("http://34.219.60.253:8000/Counter/incr").json()) # query public ip
ray.shutdown()