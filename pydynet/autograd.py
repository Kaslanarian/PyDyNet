import functools

grad_enable = True


def is_grad_enable():
    return grad_enable


def set_grad_enabled(mode: bool):
    global grad_enable
    grad_enable = mode


class no_grad:

    def __enter__(self) -> None:
        self.prev = is_grad_enable()
        set_grad_enabled(False)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        set_grad_enabled(self.prev)

    def __call__(self, func):

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with __class__():
                return func(*args, **kwargs)

        return decorate_context


class enable_grad:

    def __enter__(self) -> None:
        self.prev = is_grad_enable()
        set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        set_grad_enabled(self.prev)

    def __call__(self, func):

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with __class__():
                return func(*args, **kwargs)

        return decorate_context
