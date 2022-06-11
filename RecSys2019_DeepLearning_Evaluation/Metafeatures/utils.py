def register_func(registry):
    """
    Adds the function to the lookup dictionary "registry", accessible by function name.
    Args:
        registry:

    Returns:

    """
    def register_func_decorator(func):
        registry[func.__name__] = func
        return func
    return register_func_decorator
