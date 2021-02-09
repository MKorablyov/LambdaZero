import time
class DockingOracle:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # create actor pool

    def __call__(self, data):

        dockscores = [data["graph"] for i in range(len(data))]

        print("oracle is called on", len(data), dockscores)
        return dockscores