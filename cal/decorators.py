from typing import Protocol


class Rehashable(Protocol):

    def rehash(self) -> None:
        ...


def writer(method):
    def write_and_update_hash(self: Rehashable, *args, **kwargs):
        method(self, *args, **kwargs)
        self.rehash()
    return write_and_update_hash
