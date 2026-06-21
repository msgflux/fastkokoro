__all__ = ["FastKokoro"]


def __getattr__(name: str):
    if name == "FastKokoro":
        from fastkokoro.engine import FastKokoro

        return FastKokoro
    raise AttributeError(name)
