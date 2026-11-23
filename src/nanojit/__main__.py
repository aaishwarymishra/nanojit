from .compiler import trace


def model(x):
    return x * 2 + 3


if __name__ == "__main__":
    trace(model)
