from e3nn.non_linearities.rescaled_act import ShiftedSoftplus


v0 = {
    "max_radius": 10.,
    "number_of_basis": 50,
    "h": 128,
    "L": 2,
    "act": ShiftedSoftplus(beta=5.)
}
