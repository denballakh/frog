from __future__ import annotations
import typing as t
import collections.abc as abc
import dataclasses


@dataclasses.dataclass(slots=True, frozen=True)
class _Node:
    val: str
    next: _Node | None = None

    def __iter__(self) -> abc.Iterator[str]:
        node: _Node | None = self
        while node is not None:
            yield node.val
            node = node.next


class StringBuilder:
    __slots__: tuple[str, ...] = ('_node',)
    _node: _Node

    def __init__(self, _node: _Node = _Node('', None)) -> None:
        self._node = _node

    def add(self, s: str) -> None:
        self._node = _Node(s, self._node)

    def __iadd__(self, s: str) -> t.Self:
        self.add(s)
        return self

    def copy(self) -> t.Self:
        return self.__class__(self._node)

    def __getitem__(self, index: slice) -> t.Self:
        if index != slice(None):
            raise ValueError(f'only [::] slice allowed')
        return self.copy()

    @t.override
    def __str__(self) -> str:
        s = ''.join(reversed([*self._node]))  # is there a better way?
        self._node = _Node(s)
        return s


def example() -> None:
    # create:
    s = StringBuilder()
    # append:
    s += 'chunk'
    s.add('chunk')
    # fork:
    s2 = s.copy()
    s2 = s[::]
    _ = s2
    # build:
    _ = str(s)
