def get_heading(text: str, level: int):
    if level == 1:
        return f"""
{100 * "#"}
{f" {text} ".center(120, "#")}
{100 * "#"}
        """
    if level == 2:
        return f" {text} ".center(80, "#")
    if level == 3:
        return f" {text} ".center(60, "#")
    return text