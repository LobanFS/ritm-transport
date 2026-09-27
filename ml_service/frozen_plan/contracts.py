"""Shared contracts. Agents should change these only by agreement."""

POINT_COLUMNS = (
    "sample_id",
    "tr_id",
    "T",
    "target_stop_id",
    "target_time_begin",
    "cur_dev_s",
)
TARGET_COLUMN = "target_delay_s"
FORBIDDEN_INFERENCE_COLUMNS = frozenset(
    {TARGET_COLUMN, "target_class", "time_fact_begin"}
)
KEY_COLUMNS = ("sample_id", "tr_id", "T")
FEATURE_COLUMNS = (
    "cur_dev_s",
    "horizon_s",
    "t_hour_sin",
    "t_hour_cos",
    "t_weekday",
)
SUBMISSION_COLUMNS = ("sample_id", "prediction")
