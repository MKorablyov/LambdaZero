import ray, time, socket
from collections import Counter

@ray.remote
def some_function():
    time.sleep(0.1)
    return socket.gethostname()

ray.init(address="auto")
for i in range(10000):
    futures = [some_function.remote() for _ in range(100)]
    ips = ray.get(futures)
    print(Counter(ips))
    #print("resources", ray.cluster_resources())

