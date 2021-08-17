# general purpose
# a1.large, t2.large, t3.large, t3a.large,
# compute
# c6gn, c6gd, c6g.large, c5.large, c5ad.large, c5.large, c5d.large, c5n.large


import ray
from ray.util import placement_group

# Two "CPU"s are available.
ray.init(num_cpus=2) # address="auto" (num_cpus=2)

# Create a placement group.
pg = placement_group([{"CPU": 2}], strategy="STRICT_PACK") # {"c5_large":1}
ray.get(pg.ready())

# Now, 2 CPUs are not available anymore because they are pre-reserved by the placement group.
@ray.remote(num_cpus=2)
def f():
    return True

print(ray.get(f.options().remote()))
print(ray.get(f.options(placement_group=pg).remote()))


# Won't be scheduled because there are no 2 cpus.
#print(ray.get(f.remote()))

# Will be scheduled because 2 cpus are reserved by the placement group.


# @ray.remote(num_cpus=4)
# class Counter(object):
#     def __init__(self):
#         self.value = 0
#
#     def increment(self):
#         self.value += 1
#         return self.value
#
#
# ray.init()
# a1 = Counter.options(num_cpus=1, resources={"Custom1": 1}).remote()
# a2 = Counter.options(num_cpus=2, resources={"Custom2": 1}).remote()
# a3 = Counter.options(num_cpus=3, resources={"Custom3": 1}).remote()

