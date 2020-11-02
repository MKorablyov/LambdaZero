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
        if param.name in ['self', 'args', 'kwargs']:
            continue

        conf_name = param.name if param.name not in req_param_to_conf_param \
                               else req_param_to_conf_param[param.name]

        if conf_name in config:
            kwargs[param.name] = config[conf_name]
        elif hasattr(data, conf_name):
            kwargs[param.name] = getattr(data, conf_name)
        elif param.default is inspect.Parameter.empty:
            raise RuntimeError(
                'Could not find non-default init argument ' +
                '%s of class %s in config or data object' % (conf_name, pooling_obj.__name__)
            )

    return pooling_obj(**kwargs)

