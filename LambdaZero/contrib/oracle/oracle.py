import time
class DockingOracle:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # create actor pool

    def __call__(self, molecules):


        dockscores = [len(molecules[i].blocks) for i in range(len(molecules))]

        print("oracle is called on", len(molecules))
        return dockscores