from LambdaZero.utils import get_external_dirs
from LambdaZero.models import MPNNetDrop

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

mpnn000 = {}

mpnndrop_000 = {
    "model": MPNNetDrop,
    "model_config": {"drop_data": False, "drop_weights": False, "drop_last": True, "drop_prob": 0.1},
}
#
# mpnn001 = {
#     "trainer_config": {
#         "pow":1,
#         "use_sampler":True
#     }
# }
#
#
# mpnn002 = {
#     "trainer_config": {
#         "pow":2,
#         "use_sampler":True
#     }
# }
#
# mpnn003 = {
#     "trainer_config": {
#         "pow":3,
#         "use_sampler":True
#     }
# }
#
# mpnn004 = {
#     "trainer_config": {
#         "pow":4,
#         "use_sampler":True
#     }
# }
#
# mpnn005 = {
#     "trainer_config": {
#         "pow":5,
#         "use_sampler":True
#     }
# }
