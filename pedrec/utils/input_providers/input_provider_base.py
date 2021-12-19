from typing import Any


class InputProviderBase(object):
    """
    Base class for input providers.
    """

    def get_data(self) -> Any:
        """
        Returns one data element, e.g. the next frame from a camera or the next image from a image directory.
        :return: A data element
        """
        raise NotImplementedError
