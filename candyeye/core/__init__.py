"""Public CandyEye model and training entry points, loaded on demand."""

__all__ = ["CandyEye", "train"]


def __getattr__(name):
    if name == "CandyEye":
        from candyeye.core.candyeye import CandyEye
        return CandyEye
    if name == "train":
        from candyeye.training.trainer import train
        return train
    raise AttributeError(name)
