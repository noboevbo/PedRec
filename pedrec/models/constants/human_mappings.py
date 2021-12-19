from enum import Enum


class GENDER(Enum):
    MALE = 0
    FEMALE = 1


def get_age_from_years(age_years: int):
    if age_years <= 0:
        raise ValueError("Age can not be <= 0.")
    if age_years < 13:
        return AGE.CHILD
    if age_years < 60:
        return AGE.ADULT
    return AGE.ELDER


class AGE(Enum):
    CHILD = 0
    ADULT = 1
    ELDER = 2


def get_size_from_cm(size_cm: int):
    if size_cm <= 0:
        raise ValueError("Size can not be <= 0.")
    if size_cm < 150:
        return SIZE.SMALL
    if size_cm < 180:
        return SIZE.MEDIUM
    if size_cm < 200:
        return SIZE.LARGE
    return SIZE.XL


class SIZE(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    XL = 3


def get_bmi_from_size_weight(size_cm: int, weight_kg: int):
    if size_cm <= 0 or weight_kg <= 0:
        raise ValueError("Size or weight can not be <= 0.")
    bmi = weight_kg / ((size_cm / 100)**2)
    if bmi < 18.5:
        return BMI.UNDERWEIGHT
    if bmi < 25:
        return BMI.NORMAL
    if bmi < 30:
        return BMI.OVERWEIGHT
    return BMI.ADIPOSITAS

class BMI(Enum):
    UNDERWEIGHT = 0
    NORMAL = 1
    OVERWEIGHT = 2
    ADIPOSITAS = 3


class SKIN_COLOR(Enum):
    """
    Skin type by the Fitzpatrick classification
    """
    PALE_WHITE = 0  # Skin Type 1
    WHITE = 1  # Skin Type 2
    DARK_WHITE = 2  # Skin Type 3
    LIGHT_BROWN = 3  # Skin Type 4
    BROWN = 4  # Skin Type 5
    DARK_BROWN = 5  # Skin Type 6


sim_name_skin_color_mapping = {
    "(I) Light, pale white": SKIN_COLOR.PALE_WHITE,
    "(II) White, fair": SKIN_COLOR.WHITE,
    "(III) Medium, white to light brown": SKIN_COLOR.DARK_WHITE,
    "(IV) Olive, moderate brown": SKIN_COLOR.LIGHT_BROWN,
    "(V) Brown, dark brown": SKIN_COLOR.BROWN,
    "(VI) Very dark brown to black": SKIN_COLOR.DARK_BROWN
}

class MOVEMENT(Enum):
    STAND = 0
    WALK = 1
    JOG = 2
    RUN = 3


class MOVEMENT_SPEED(Enum):
    NONE = 0
    SLOW = 1
    NORMAL = 2
    FAST = 3
