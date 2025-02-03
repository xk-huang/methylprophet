"""
Analysis is copied from https://github.com/open-mmlab/mmengine/tree/9124ebf7a285aa785de77ad567aba06b5f276032/mmengine/analysis
"""

from src.utils.analysis import (
    ActivationAnalyzer,
    FlopAnalyzer,
    activation_count,
    flop_count,
    get_model_complexity_info,
    parameter_count,
    parameter_count_table,
)
from src.utils.io import prepare_version_dir
