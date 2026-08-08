"""Pins current (buggy) extraction for backlog defects E4, E5, E13.
E12 was fixed by Task 2.1; the construct below stays for snapshot coverage of
the now-correct output.

Task 0.1 freezes the wrong output rather than fixing it -- see the hardening
plan, Phase 0.
"""

__all__ = ["summarize", "Worker"]  # E13: __all__ entries never become export rows


def summarize(items):
    return list(items)


def call_with_generator(items):
    return summarize(x for x in items)  # E4: generator-sole-argument call is dropped


class Base:
    pass


class Meta(type):
    pass


class Worker(Base, metaclass=Meta):  # E12 (fixed, Task 2.1): keyword_argument
    # wrapper is skipped now; only `Base` becomes an inheritance row.
    pass


class Config:
    TIMEOUT = 30

    def apply(self, config):
        config.TIMEOUT = 5  # E5: member-attribute write is swallowed by the left exclusion
        return config.TIMEOUT  # E5: member-attribute read never appears
