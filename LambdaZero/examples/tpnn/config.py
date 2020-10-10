tpnn_4_64_0_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(64, 0, 0)]] * 3 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_8_64_0_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(64, 0, 0)]] * 7 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_16_64_0_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(64, 0, 0)]] * 15 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_4_64_01_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(16, 0, 0), (16, 1, 0)]] * 3 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_8_64_01_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(16, 0, 0), (16, 1, 0)]] * 7 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_16_64_01_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(16, 0, 0), (16, 1, 0)]] * 15 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_4_64_012_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(16, 0, 0), (11, 1, 0), (3, 2, 0)]] * 3 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_8_64_012_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(16, 0, 0), (11, 1, 0), (3, 2, 0)]] * 7 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_16_64_012_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(16, 0, 0), (11, 1, 0), (3, 2, 0)]] * 15 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_4_128_0_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(128, 0, 0)]] * 3 + [[(128, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_4_128_01_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(32, 0, 0), (32, 1, 0)]] * 3 + [[(128, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_8_128_01_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(32, 0, 0), (32, 1, 0)]] * 7 + [[(128, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_4_64_01234_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + [[(11, 0, 0), (5, 1, 0), (3, 2, 0), (2, 3, 0), (1, 4, 0)]] * 3 + [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}


tpnn_4_64_0123456_set2set = {
    "trainer_config":
    {
        "model_config":
        {
            "representations": [[(23, 0, 0)]] + 
			       [[(10, 0, 0), (3, 1, 0), (1, 2, 0), (1, 3, 0), (1, 4, 0), (1, 5, 0), (1, 6, 0)]] + 
			       [[(11, 0, 0), (5, 1, 0), (3, 2, 0), (2, 3, 0), (1, 4, 0)]] + 
			       [[(10, 0, 0), (3, 1, 0), (1, 2, 0), (1, 3, 0), (1, 4, 0), (1, 5, 0), (1, 6, 0)]] + 
			       [[(64, 0, 0)]],
            "use_set2set": True
        }
    }
}
