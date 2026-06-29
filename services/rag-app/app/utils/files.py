from datetime import datetime


def _coerce_filename_value(name) -> str:
    if isinstance(name, dict):
        for key in ("source", "filename", "name", "title"):
            value = name.get(key)
            if value:
                return str(value)
        return ""
    return str(name or "")


def safe_filename(name: str) -> str:
    value = _coerce_filename_value(name).strip().replace("\\", "/")
    value = value.split("/")[-1]
    value = "".join([char for char in value if char.isalnum() or char in (".", "_", "-", " ")])
    return value or ("file_" + datetime.now().strftime("%Y%m%d_%H%M%S"))


def filename_stem(name: str) -> str:
    value = _coerce_filename_value(name).strip().replace("\\", "/").split("/")[-1]
    if "." in value:
        value = ".".join(value.split(".")[:-1])
    return value


def normalize_filename_for_match(name: str) -> str:
    value = _coerce_filename_value(name).strip().replace("\\", "/")
    return value.split("/")[-1]
