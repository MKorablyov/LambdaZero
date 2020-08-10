from LambdaZero.utils import get_external_dirs
import LambdaZero.models

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

mpnn000 = {}
mpnn000_L2 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
        },
        "model": LambdaZero.models.MPNNet
}

mpnn_log_L2 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":True,
                    "mode":"log",
                    "loss": "L2"
                    },
        "model": LambdaZero.models.MPNNet
}

mpnn_log_NT_L2 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":False,
                    "mode":"log",
                    "loss": "L2"
                    },
        "model": LambdaZero.models.MPNNet
}

mpnn001_L2 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}


mpnn002_L2 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn003_L2 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn004_L2 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn005_L2 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn000_NT_L2 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
        },
        "model": LambdaZero.models.MPNNet
}

mpnn001_NT_L2 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}


mpnn002_NT_L2 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn003_NT_L2 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn004_NT_L2 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn005_NT_L2 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.MPNNet
}


mpnn000_L1 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
        },
        "model": LambdaZero.models.MPNNet
}


mpnn_log_L1 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":True,
                    "mode":"log",
                    "loss": "L1"
                    },
        "model": LambdaZero.models.MPNNet
}

mpnn_log_NT_L1 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":False,
                    "mode":"log",
                    "loss": "L1"
                    },
        "model": LambdaZero.models.MPNNet
}

mpnn001_L1 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}


mpnn002_L1 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn003_L1 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn004_L1 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn005_L1 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn000_NT_L1 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
        },
        "model": LambdaZero.models.MPNNet
}

mpnn001_NT_L1 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}


mpnn002_NT_L1 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn003_NT_L1 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn004_NT_L1 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn005_NT_L1 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.MPNNet
}

mpnn020 = {
                "trainer_config":{
                                "use_sampler":True,
                                "use_tail":True,
                                "mode":"pow",
                                "loss": "L2",
                                "pow":20
                },
        "model": LambdaZero.models.MPNNet
}


dime000_L2 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
        },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime_log_L2 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":True,
                    "mode":"log",
                    "loss": "L2"
                    },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime_log_NT_L2 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":False,
                    "mode":"log",
                    "loss": "L2"
                    },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime001_L2 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}


dime002_L2 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime003_L2 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime004_L2 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime005_L2 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime000_NT_L2 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
        },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime001_NT_L2 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}


dime002_NT_L2 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime003_NT_L2 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime004_NT_L2 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime005_NT_L2 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L2"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}


dime000_L1 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
        },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}


dime_log_L1 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":True,
                    "mode":"log",
                    "loss": "L1"
                    },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime_log_NT_L1 = {
        "trainer_config": {
                    "use_sampler":True,
                    "use_tail":False,
                    "mode":"log",
                    "loss": "L1"
                    },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

mpnn001_L1 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}


dime002_L1 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime003_L1 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime004_L1 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime005_L1 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":True,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime000_NT_L1 = {
        "trainer_config":{
            "use_sampler":False,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
        },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime001_NT_L1 = {
        "trainer_config": {
            "pow":1,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}


dime002_NT_L1 = {
        "trainer_config": {
            "pow":2,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime003_NT_L1 = {
        "trainer_config": {
            "pow":3,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime004_NT_L1 = {
        "trainer_config": {
            "pow":4,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime005_NT_L1 = {
        "trainer_config": {
            "pow":5,
            "use_sampler":True,
            "use_tail":False,
            "mode":"pow",
            "loss": "L1"
            },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}

dime020 = {
                "trainer_config":{
                                "use_sampler":True,
                                "use_tail":True,
                                "mode":"pow",
                                "loss": "L2",
                                "pow":20
                },
        "model": LambdaZero.models.DimeNet(hidden_channels=128, out_channels=1, num_blocks=6,
                        num_bilinear=8, num_spherical=7, num_radial=6,
                        cutoff=5.0, envelope_exponent=5, num_before_skip=1,
                        num_after_skip=2, num_output_layers=3)
}
