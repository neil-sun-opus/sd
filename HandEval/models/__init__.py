from importlib import import_module

_REGISTRY = {}

def register(name):
    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls
    return wrapper

def get_model_class(name):
    if name not in _REGISTRY:
        # 惰性导入，避免无用依赖
        import_module(f"models.{name}_wrapper")
    return _REGISTRY[name]