
def array_may_13(args):
    base = {'replay_mode': 'online',
            'sample_prob': 0.9,
            'mbsize': 8,
            'nemb': 50,
    }

    all_hps = [
        {**base, 'run': 300},
        {**base, 'run': 301, 'sample_prob': 0.75},
        {**base, 'run': 302, 'sample_prob': 0.75, 'bootstrap_tau': 0.1},
        {**base, 'run': 303, 'sample_prob': 1, 'max_blocks': 3, 'min_blocks': 1},
        {**base, 'run': 304, 'sample_prob': 1, 'max_blocks': 3, 'min_blocks': 1, 'mbsize': 32},
        {**base, 'run': 305, 'sample_prob': 1, 'max_blocks': 3, 'min_blocks': 1, 'model_version': 'v3'},
        {**base, 'run': 306, 'sample_prob': 1, 'max_blocks': 3, 'min_blocks': 1, 'model_version': 'v4'},
        {**base, 'run': 307, 'sample_prob': 1, 'max_blocks': 3, 'min_blocks': 1, 'model_version': 'v4',
         'num_conv_steps': 6, 'nemb': 128},
        {**base, 'run': 308, 'sample_prob': 1, 'max_blocks': 2, 'min_blocks': 1, 'model_version': 'v4',
         'num_conv_steps': 6, 'nemb': 128},
        {**base, 'run': 309, 'sample_prob': 1, 'max_blocks': 2, 'min_blocks': 1, 'model_version': 'v4',
         'num_conv_steps': 6, 'nemb': 256, 'mbsize': 32},
        {**base, 'run': 310, 'sample_prob': 1, 'max_blocks': 2, 'min_blocks': 1, 'model_version': 'v4',
         'num_conv_steps': 6, 'nemb': 128, 'log_reg_c': 1e-2, 'mbsize': 2},
        {**base, 'run': 311, 'sample_prob': 1, 'max_blocks': 2, 'min_blocks': 1, 'model_version': 'v4',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 2},
        {**base, 'run': 312, 'sample_prob': 1, 'max_blocks': 2, 'min_blocks': 1, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 2},
        {**base, 'run': 313, 'sample_prob': 1, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4, 'learning_rate': 5e-4},
        {**base, 'run': 314, 'sample_prob': 1, 'max_blocks': 2, 'min_blocks': 1, 'model_version': 'v6',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 2},
        {**base, 'run': 315, 'sample_prob': 1, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4, 'learning_rate': 5e-4,
         'reward_exp': 4},
        {**base, 'run': 316, 'sample_prob': 1, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4, 'learning_rate': 5e-4,
         'reward_exp': 8},
        {**base, 'run': 317, 'sample_prob': 1, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4,
         'reward_exp': 8},
        {**base, 'run': 318, 'sample_prob': 1, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4, 'learning_rate': 5e-4,
         'reward_exp': 8, 'clip_grad': 0.5}, #norm clip
        {**base, 'run': 319, 'sample_prob': 1, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4, 'learning_rate': 5e-4,
         'reward_exp': 8, 'clip_grad': 0.5}, # value clip
        {**base, 'run': 320, 'sample_prob': 0.75, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4, 'learning_rate': 5e-4,
         'reward_exp': 8, 'clip_grad': 0.5},
        {**base, 'run': 321, 'sample_prob': 1, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4, 'learning_rate': 5e-4,
         'reward_exp': 8, 'clip_loss': 1},
        {**base, 'run': 322, 'sample_prob': 1, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4, 'learning_rate': 5e-4,
         'reward_exp': 8, 'clip_loss': 10},
        {**base, 'run': 323, 'sample_prob': 1, 'max_blocks': 8, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 6, 'nemb': 256, 'log_reg_c': 1e-2, 'mbsize': 4, 'learning_rate': 5e-4,
         'reward_exp': 8, 'clip_loss': 100},
        {**base, 'run': 324, 'sample_prob': 1, 'max_blocks': 4, 'min_blocks': 2, 'model_version': 'v5',
         'num_conv_steps': 1, 'nemb': 2, 'log_reg_c': 1e-2, 'mbsize': 1, 'learning_rate': 5e-4,
         'reward_exp': 4, 'num_iterations': 10},

    ]
    base2 = {'replay_mode': 'online',
             'sample_prob': 1,
             'mbsize': 4,
             'nemb': 256,
             'max_blocks': 8,
             'min_blocks': 2,
             'model_version': 'v5',
             'num_conv_steps': 6,
             'learning_rate': 5e-4,
             'num_iterations': 30000,
             'reward_exp': 4,
             'reward_norm': 1,
    }
    all_hps += [
        ##
        {**base2, 'run': 325, 'log_reg_c': 1e-2},
        {**base2, 'run': 326, 'log_reg_c': 1e-2, 'weight_decay': 1e-5},
        {**base2, 'run': 327, 'log_reg_c': 1e-2, 'weight_decay': 1e-5, 'bootstrap_tau': 0.01},
        {**base2, 'run': 328, 'log_reg_c': 1e-2, 'bootstrap_tau': 0.01, 'learning_rate': 2.5e-4},
        {**base2, 'run': 329, 'log_reg_c': 1e-4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 8},
        {**base2, 'run': 330, 'log_reg_c': 1e-4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 8, 'floatX': 'float32', 'mbsize': 16},
        {**base2, 'run': 331, 'log_reg_c': 1e-2, 'balanced_loss': False},
        {**base2, 'run': 332, 'log_reg_c': 1e-2},
        {**base2, 'run': 333, 'log_reg_c': 1e-2, 'opt_beta2': 0.999},
        {**base2, 'run': 334, 'log_reg_c': 1e-2, 'opt_beta2': 0.9995, 'opt_beta':0.95},
        {**base2, 'run': 335, 'log_reg_c': 1e-2, 'opt_beta2': 0.999, 'opt_beta':0.95,
         'learning_rate': 1e-4},
        {**base2, 'run': 336, 'log_reg_c': 1e-2, 'opt_beta2': 0.999, 'opt_beta':0.95,
         'learning_rate': 1e-4, 'random_action_prob': 0.01},
        ###
        {**base2, 'run': 337, 'log_reg_c': 1e-2, 'opt_beta2': 0.999,
         'random_action_prob': 0.01, 'opt_epsilon': 1e-2}, #  this is best of 337-339
        {**base2, 'run': 338, 'log_reg_c': 1e-2, 'opt_beta2': 0.999,
         'opt_epsilon': 1e-2}, # this on its own not so much?
        {**base2, 'run': 339, 'log_reg_c': 1e-2, 'opt_beta2': 0.999,
         'random_action_prob': 0.01}, # This is good
        # includes nblocks attr reg
        {**base2, 'run': 340, 'log_reg_c': 1e-5, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 6, 'random_action_prob': 0.05},
        # includes / (s.nblocks * max_blocks) denominator
        {**base2, 'run': 341, 'log_reg_c': 1e-4, 'reward_norm': 8, 'learning_rate': 1e-5,
         'num_conv_steps': 6, 'random_action_prob': 0.05},
        # no denominator
        {**base2, 'run': 342, 'log_reg_c': 1e-4, 'reward_norm': 8, 'learning_rate': 1e-5,
         'num_conv_steps': 6, 'random_action_prob': 0.05},
        {**base2, 'run': 343, 'log_reg_c': 1e-4, 'reward_norm': 8, 'learning_rate': 1e-5,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999,},
        {**base2, 'run': 344, 'log_reg_c': 1e-2, 'reward_norm': 8, 'learning_rate': 1e-5,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999, 'bootstrap_tau': 0.1},
        ########
        {**base2, 'run': 345, 'log_reg_c': 1, 'reward_norm': 8, 'learning_rate': 1e-5,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999, 'bootstrap_tau': 0.1,
         'num_iterations': 50, 'R_min': 0.1},
        {**base2, 'run': 346, 'log_reg_c': 1, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1},
        {**base2, 'run': 347, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1}, # seems to be a good log_reg_c
        {**base2, 'run': 348, 'log_reg_c': (2/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1}, # cuz this one is bad
        {**base2, 'run': 349, 'log_reg_c': 0, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1}, # and this one too
        {**base2, 'run': 350, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10}, # this was a good run... and then it died?
        {**base2, 'run': 351, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.9995, 'mbsize': 8,
         'R_min': 0.1, 'leaf_coef': 10},
        {**base2, 'run': 352, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'include_nblocks': False}, # the nblocks seems to help
        {**base2, 'run': 353, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 6, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10}, # rerun of 350
        {**base2, 'run': 354, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 20, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10},
        {**base2, 'run': 355, 'log_reg_c': (0.5/8)**8, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 20, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.5, 'leaf_coef': 10, 'reward_exp': 8},
        {**base2, 'run': 356, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 20, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'nemb': 64, 'mbsize':32}, # <nemb, >mbsize
        {**base2, 'run': 357, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10}, # 350 + lr 5e-4 + ncs
        ### block arch??
        {**base2, 'run': 358, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10,
         'repr_type': 'block_graph', 'include_nblocks': False}, # block arch
        {**base2, 'run': 359, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'mbsize': 64,
         'repr_type': 'block_graph', 'include_nblocks': False, 'nemb': 128}, # >>mbsize, <nemb
        # ^ learns the same distribution, much much slower.
        {**base2, 'run': 360, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10,
         'repr_type': 'block_graph', 'include_nblocks': False}, # higher LR
        {**base2, 'run': 361, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'mbsize': 2, 'nemb': 256+128, 'num_iterations':int(60e3),
         'repr_type': 'block_graph', 'include_nblocks': False},# more iters, <mbsize, >nemb
        {**base2, 'run': 362, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 8,
         'repr_type': 'block_graph', 'include_nblocks': False}, # higher LR, reward exp 8

        {**base2, 'run': 363, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 10,
         'repr_type': 'block_graph', 'include_nblocks': False}, # higher LR, reward exp 10
        # good stuff

        {**base2, 'run': 364, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 12,
         'repr_type': 'block_graph', 'include_nblocks': False}, # higher LR, reward exp 12
        {**base2, 'run': 365, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 20, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 8,
         'repr_type': 'block_graph', 'include_nblocks': False}, # higher LR, reward exp 8, 20 conv @362

        # @ 363
        {**base2, 'run': 366, 'log_reg_c': (0.1/8)**6, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False}, # lower log_reg_c **6
        # ^ finds better molecules but losses have a lot more variance, could be less stable
        {**base2, 'run': 367, 'log_reg_c': (0.1/8)**8, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False}, # lower log_reg_c **8
        # ^ finds better molecules but losses have a lot more variance, could be less stable

        {**base2, 'run': 368, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.01, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False}, # lower random_action_prob
        # ^ better molecules and more unique, interesting?
        {**base2, 'run': 369, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.1, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False}, # higher random_action_prob
        # ^ worse molecules and less unique?? strange

        {**base2, 'run': 370, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.1, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 5, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False}, # leaf coef 5
        # ^ worse molecules, slightly higher leaf loss
        {**base2, 'run': 371, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.1, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 1, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False}, # leaf coef 1
        # ^ worse molecules, slightly higher leaf loss, might just be
        # chance but produces much larger (nblocks) molecules

        {**base2, 'run': 372, 'log_reg_c': (0.1/8)**6, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False,
         'num_iterations': 100000,}, # long run, what happens?
        # no divergence, keeps generating good mols but unicity slowly
        # goes down (still above 80%). Loss does not go to 0

        {**base2, 'run': 373, 'log_reg_c': (0.1/8)**8, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.01, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 10, 'nemb': 512,
         'repr_type': 'block_graph', 'include_nblocks': False}, # nemb 512, random 0.01, low lrc
        # slightly better mols, not that slower but takes lots of VRAM, perhaps not worth it
        {**base2, 'run': 374, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.05, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 10, 'nemb': 256+128,
         'repr_type': 'block_graph', 'include_nblocks': False}, # >nemb
        # worse? I mean, not better.

        {**base2, 'run': 375, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.1, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 20, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False}, # leaf coef 20 (vs 10 @ 363)
        {**base2, 'run': 376, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 5e-4,
         'num_conv_steps': 10, 'random_action_prob': 0.1, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 50, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False}, # leaf coef 50
        {**base2, 'run': 377, 'log_reg_c': (0.1/8)**4, 'reward_norm': 8, 'learning_rate': 1e-3,
         'num_conv_steps': 10, 'random_action_prob': 0.1, 'opt_beta2': 0.999,
         'R_min': 0.1, 'leaf_coef': 10, 'reward_exp': 10, 'model_version': 'v4',
         'repr_type': 'block_graph', 'include_nblocks': False}, # lr 1e-3 (from 5e-4)
    ]

    base3 = {'replay_mode': 'online',
             'num_iterations': int(1e6/4),
             'mbsize': 4,
             'sample_prob': 1,
             'nemb': 256,
             'max_blocks': 8,
             'min_blocks': 2,
             'model_version': 'v4',
             'repr_type': 'block_graph',
             'num_conv_steps': 10,
             'opt_beta2': 0.999,
             'learning_rate': 5e-4,
             'log_reg_c': (0.1/8)**8,
             'reward_exp': 10,
             'reward_norm': 8,
             'R_min': 0.1,
             'random_action_prob': 0.01,
             'leaf_coef': 20,
    }
    all_hps += [
        {**base3, 'run': 378},
        {**base3, 'run': 379},
        {**base3, 'run': 380},
        {**base3, 'run': 381, 'do_wrong_thing': True},
        {**base3, 'run': 382, 'replay_mode': 'prioritized', 'sample_prob': 0.75},
        {**base3, 'run': 383},
    ]

    return all_hps


def array_aug_7(args):

    base3 = {'replay_mode': 'online',
             'num_iterations': int(2e5/4),
             'mbsize': 4,
             'sample_prob': 1,
             'nemb': 256,
             'max_blocks': 8,
             'min_blocks': 2,
             'model_version': 'v4',
             'repr_type': 'block_graph',
             'num_conv_steps': 10,
             'opt_beta2': 0.999,
             'learning_rate': 5e-4,
             'log_reg_c': (0.1/8)**8,
             'reward_exp': 10,
             'reward_norm': 1,
             'R_min': 0.01,
             'random_action_prob': 0.01,
             'leaf_coef': 20,
    }
    all_hps = [
        {**base3, 'run': 1000},
        {**base3, 'run': 1001, 'reward_exp': 20},
        {**base3, 'run': 1002, 'reward_exp': 12},
        {**base3, 'run': 1003, 'reward_exp': 8},
        {**base3, 'run': 1004, 'reward_exp': 6},
        {**base3, 'run': 1005, 'reward_exp': 8, 'reward_norm': 25},
        {**base3, 'run': 1006, 'reward_exp': 10, 'reward_norm': 25},
        {**base3, 'run': 1007, 'reward_exp': 6, 'reward_norm': 25},
        {**base3, 'run': 1008, 'reward_exp': 6, 'reward_norm': 20},
    ]

    return all_hps
