from e3nn.point.gate import Gate
from e3nn.non_linearities.rescaled_act import Softplus, sigmoid

v0 = {
    "type": Gate,
    "config": {
        "scalar_act": Softplus(beta=5.),
        "tensor_act": sigmoid
    }
}
