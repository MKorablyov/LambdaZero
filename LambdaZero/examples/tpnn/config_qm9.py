from e3nn.radial import CosineBasisModel
from e3nn.point.gate import Gate
from e3nn.non_linearities.rescaled_act import sigmoid, Softplus, ShiftedSoftplus
from functools import partial


tpnn_l1net_v0 = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(128, 0, 0)]] +
                               [[(64, 0, 0), (21, 1, 0)]] * 4 +
                               [[(64, 0, 0)]],
            "radial_model": partial(CosineBasisModel, max_radius=10., number_of_basis=50, h=128, L=2, act=ShiftedSoftplus(beta=5.)),
            "gate": partial(Gate, scalar_act=Softplus(beta=5.), tensor_act=sigmoid)
        }
    }
}
