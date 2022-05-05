# Adds the function to the lookup dictionary "registry", accessible by function name.
def register_func(registry):
    def register_func_decorator(func):
        registry[func.__name__] = func
        return func
    return register_func_decorator
