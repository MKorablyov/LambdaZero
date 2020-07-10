from LambdaZero.utils import get_external_dirs


datasets_dir, programs_dir, summaries_dir = get_external_dirs()

vanilla_mpnn_L2 = {
    "trainer_config": {
        "use_sampler":False,
        "loss": "L2"
    }
}

vanilla_mpnn_L1 = {
    "trainer_config": {
        "use_sampler":False,
        "loss": "L1"
    }
}


neg_mpnn001_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow": -1,
        "use_sampler":True
    }
}


mpnn000_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow": 0,
        "use_sampler":True
    }
}


mpnn001_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow":1,
        "use_sampler":True
    }
}

mpnn001_L1 = {
    "trainer_config": {
        "loss": "L1",
        "pow":1,
        "use_sampler":True
    }
}

mpnn002_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow":2,
        "use_sampler":True
    }
}

mpnn002_L1 = {
    "trainer_config": {
        "loss": "L1",
        "pow":2,
        "use_sampler":True
    }
}


mpnn003_L2 = {
    "trainer_config": {
        "loss": "L2",
        "pow":3,
        "use_sampler":True
    }
}

mpnn003_L1 = {
    "trainer_config": {
        "loss": "L1",
        "pow":3,
        "use_sampler":True
    }
}

mpnn004_L2 = {
    "trainer_config": {
        "pow":4,
        "use_sampler":True,
        "loss": "L2"
    }
}

mpnn004_L1 = {
    "trainer_config": {
        "pow":4,
        "use_sampler":True,
        "loss": "L1"
    }
}

mpnn005_L2 = {
    "trainer_config": {
        "pow":5,
        "use_sampler":True,
        "loss": "L2"
    }
}

mpnn005_L1 = {
    "trainer_config": {
        "pow":5,
        "use_sampler":True,
        "loss": "L1"
    }
}

mpnn020_L2 = {
    "trainer_config": {
        "pow":20,
        "use_sampler":True,
        "loss": "L2"
    }
}

vanilla_dime_L2 = {
    "trainer_config": {
        "model": "dime",
        "use_sampler":False,
        "loss": "L2",
        "b_size": 16
    }
}

