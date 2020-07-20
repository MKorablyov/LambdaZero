from LambdaZero.utils import get_external_dirs


datasets_dir, programs_dir, summaries_dir = get_external_dirs()

mpnn000_L2 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
        }
}

mpnn_log_L2 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":True,
                    "mode":"log",
                    "loss": "L2"
                    }
}

mpnn_log_NT_L2 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":False,
                    "mode":"log",
                    "loss": "L2"
                    }
}

mpnn001_L2 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            }
}


mpnn002_L2 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            }
}

mpnn003_L2 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            }
}

mpnn004_L2 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            }
}

mpnn005_L2 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            }
}

mpnn000_NT_L2 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
        }
}

mpnn001_NT_L2 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            }
}


mpnn002_NT_L2 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            }
}

mpnn003_NT_L2 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            }
}

mpnn004_NT_L2 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            }
}

mpnn005_NT_L2 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            }
}


mpnn000_L1 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
        }
}


mpnn_log_L1 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":True,
                    "mode":"log",
                    "loss": "L1"
                    }
}

mpnn_log_NT_L1 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":False,
                    "mode":"log",
                    "loss": "L1"
                    }
}

mpnn001_L1 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            }
}


mpnn002_L1 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            }
}

mpnn003_L1 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            }
}

mpnn004_L1 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            }
}

mpnn005_L1 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            }
}

mpnn000_NT_L1 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
        }
}

mpnn001_NT_L1 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            }
}


mpnn002_NT_L1 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            }
}

mpnn003_NT_L1 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            }
}

mpnn004_NT_L1 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            }
}

mpnn005_NT_L1 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            }
}

mpnn020 = {
                "trainer_config":{
                                "use_sampler":True,
                                "use_tail":True,
                                "mode":"pow",
                                "loss": "L2",
                                "pow":20
                }
}
