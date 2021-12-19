from typing import List, OrderedDict

from pedrec.models.constants.generics import T


def get_without_duplicates(x: List[T]) -> List[T]:
    """
    Removes duplicated entries from a list and returns as a new list.
    """
    return list(OrderedDict.fromkeys(x))
