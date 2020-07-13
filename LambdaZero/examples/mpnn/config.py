from LambdaZero.utils import get_external_dirs


datasets_dir, programs_dir, summaries_dir = get_external_dirs()

log_mpnn_L2 = {
    "trainer_config": {
        "use_sampler":True,
        "loss": "L2",
        "mode": "log",
        "without_tail": False
    }
}
log_mpnn_L1 = {
    "trainer_config": {
        "use_sampler":True,
        "loss": "L1",
        "mode": "log",
        "without_tail": False
    }
}

vanilla_mpnn_L2_NT = {
    "trainer_config": {
        "use_sampler":False,
        "loss": "L2",
        "mode": "pow",
        "without_tail": True
    }
}

vanilla_mpnn_L1_NT = {
    "trainer_config": {
        "use_sampler":False,
        "loss": "L1",
        "mode": "pow",
        "without_tail": True
    }
}

vanilla_mpnn_L2 = {
    "trainer_config": {
        "use_sampler":False,
        "loss": "L2",
        "mode": "pow",
        "without_tail": False
    }
}

vanilla_mpnn_L1 = {
    "trainer_config": {
        "use_sampler":False,
        "loss": "L1",
        "mode": "pow",
        "without_tail": False
    }
}

neg_mpnn001_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow": -1,
        "use_sampler":True,
        "mode": "pow",
        "without_tail": False
    }
}


mpnn000_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow": 0,
        "use_sampler":True,
        "mode": "pow",
        "without_tail": False
    }
}


mpnn001_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow":1,
        "use_sampler":True,
        "mode": "pow",
        "without_tail": False
    }
}

mpnn001_L1 = {
    "trainer_config": {
        "loss": "L1",
        "pow":1,
        "use_sampler":True,
        "mode": "pow",
        "without_tail": False
    }
}

mpnn002_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow":2,
        "use_sampler":True,
        "mode": "pow",
        "without_tail": False
    }
}

mpnn002_L1 = {
    "trainer_config": {
        "loss": "L1",
        "pow":2,
        "use_sampler":True,
        "mode": "pow",
        "without_tail": False
    }
}


mpnn003_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow":3,
        "use_sampler":True,
        "mode": "pow",
        "without_tail": False
    }
}

mpnn003_L1 = {
    "trainer_config": {
        "loss": "L1",
        "pow":3,
        "use_sampler":True,
        "mode": "pow",
        "without_tail": False
    }
}

mpnn004_L2 = {
    "trainer_config": {
        "pow":4,
        "use_sampler":True,
        "loss": "L2",
        "mode": "pow",
        "without_tail": False
    }
}

mpnn004_L1 = {
    "trainer_config": {
        "pow":4,
        "use_sampler":True,
        "loss": "L1",
        "mode": "pow",
        "without_tail": False
    }
}

mpnn005_L2 = {
    "trainer_config": {
        "pow":5,
        "use_sampler":True,
        "loss": "L2",
        "mode": "pow",
        "without_tail": False
    }
}

mpnn005_L1 = {
    "trainer_config": {
        "pow":5,
        "use_sampler":True,
        "loss": "L1",
        "mode": "pow",
        "without_tail": False
    }
}

mpnn020_L2 = {
    "trainer_config": {
        "pow":20,
        "use_sampler":True,
        "loss": "L2",
        "mode": "pow",
        "without_tail": False
    }
}

vanilla_dime_L2 = {
    "trainer_config": {
        "model": "dime",
        "use_sampler":False,
        "loss": "L2",
        "mode": "pow",
        "without_tail": False
    }
}

