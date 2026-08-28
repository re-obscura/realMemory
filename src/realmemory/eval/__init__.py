"""Бенчмарки realMemory."""
from importlib import import_module


def __getattr__(name):
    if name == "run":
        return import_module("realmemory.eval.bench_recall").run
    raise AttributeError(name)


__all__ = ["run"]  # noqa: F822 - имя резолвится лениво через __getattr__ выше
