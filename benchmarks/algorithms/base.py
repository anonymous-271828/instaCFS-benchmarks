

class BaseMBAlgorithm:
    def __init__(
            self,
            association_fn,
            **kwargs):
        self.association_fn = association_fn
        self._kwargs = kwargs

    def associate(self, X, T, data, **kwargs):
        return self.association_fn(X, T, **kwargs)