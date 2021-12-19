from typing import Union


def get_seconds_as_string(seconds: Union[int, float], template: str = "{:d}:{:02d}:{:02d}"):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return template.format(h, m, s)
