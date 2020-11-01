import inspect

def construct_pooling(pooling_obj, data, config, req_param_to_conf_param={}):
    if pooling_obj is None:
        return None

    if inspect.isfunction(pooling_obj):
        return pooling_obj

    # Not a function, must be a class
    assert inspect.isclass(pooling_obj)
    kwargs = {}
    sig = inspect.signature(pooling_obj.__init__)
    for param in sig.parameters.values():
        if param.name == 'self':
            continue

        mapped_name = None if param.name not in req_param_to_conf_param \
                           else req_param_to_conf_param[param.name]

        if param.name in config:
            kwargs[param.name] = config[param.name]
        elif mapped_name is not None and mapped_name in config:
            kwargs[param.name] = config[param.name]
        elif hasattr(data, param.name):
            kwargs[param.name] = getattr(data, param.name)
        elif param.value is inspect.Parameter.empty:
            raise RuntimeError(
                'Could not find non-default init argument %s of class %s ' +
                'in config or data object' % (arg, pooling_obj.__name__)
            )

    return pooling_obj(**kwargs)

