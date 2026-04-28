import enum


if not hasattr(enum, "StrEnum"):
    class _CompatStrEnum(str, enum.Enum):
        pass

    enum.StrEnum = _CompatStrEnum

StrEnum = enum.StrEnum


class SpecialToken(StrEnum):
    DOB = "MEDS_BIRTH"
    DEATH = "MEDS_DEATH"
    TIMELINE_END = "TIMELINE_END"

    ADMISSION = "HOSPITAL_ADMISSION"
    DISCHARGE = "HOSPITAL_DISCHARGE"

    ICU_ADMISSION = "ICU_ADMISSION"
    ICU_DISCHARGE = "ICU_DISCHARGE"
    ED_ADMISSION = "ED_REGISTRATION"
    ED_DISCHARGE = "ED_OUT"
    SOFA = "SOFA"


STATIC_DATA_FN = "static_data.pickle"
