"""Nested definition shapes."""

IMPORTANT = 1


def module_level(value):
    def inner(other):
        return other + 1

    return inner(value)


class Outer:
    CONSTANT = 2

    def method(self, value):
        def closure():
            return value

        return closure

    class Inner:
        def deep_method(self):
            return self.CONSTANT


TRAILING = Outer()


def after_class():
    return TRAILING
