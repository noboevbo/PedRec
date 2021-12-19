from dataclasses import dataclass
from typing import Tuple, List

from pedrec.models.constants.generics import NUM


@dataclass
class ImageSize(object):
    width: int
    height: int

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, ImageSize):
            return self.width == other.width and \
                   self.height == other.height
        return False


@dataclass
class Color(object):
    r: int
    g: int
    b: int
    alpha: float = 255

    @property
    def tuple_rgb(self) -> Tuple[int, int, int]:
        """
        Returns a RGB Tuple
        :return: RGB Tuple
        """
        return self.r, self.g, self.b

    @property
    def tuplef_rgba(self) -> Tuple[float, float, float, float]:
        """
        Returns a RGB Tuple
        :return: RGB Tuple
        """
        return self.r / 255, self.g / 255, self.b / 255, self.alpha / 255

    @property
    def tuple_bgr(self) -> Tuple[int, int, int]:
        """
        Returns a BGR Tuple
        :return: BGR Tuple, e.g. for use in OpenCV functions
        """
        return self.b, self.g, self.r

    @property
    def rgba_float_list(self) -> List[float]:
        return [self.r / 255, self.g / 255, self.b / 255, self.alpha / 255]

    @property
    def hex(self) -> str:
        """
        Returns the hex value of the color
        :return: hex value of the color
        """
        return '#%02x%02x%02x' % (self.r, self.g, self.b)

    @classmethod
    def from_hex(cls, hex: str):
        """
        Calculates a RGB value from hex value and returns a RGB Color.
        :param hex: The hex color string
        :return: RGB Color
        """
        rgb_tuple = tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
        return Color(r=rgb_tuple[0], g=rgb_tuple[1], b=rgb_tuple[2])

    @classmethod
    def is_color_channel_value(cls, value: int):
        """
        Checks if a given integer value is a valid color channel value
        :param value: The value which should be used as a color channel value
        :return: True if it is a valid color channel value, else false
        """
        return 0 <= value <= 255


@dataclass
class Triangle:
    a: float
    b: float
    c: float
    alpha_rad: float
    beta_rad: float
    gamma_rad: float


@dataclass
class Vec2D:
    x: NUM
    y: NUM


@dataclass
class Vec3D(Vec2D):
    z: NUM
