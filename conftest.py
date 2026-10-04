import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def isolated_files(tmp_path, monkeypatch):
    """Тесты не должны создавать config.json, calibration.json и журналы в папке проекта."""
    import config as C
    for name, fname in (("CONFIG_PATH", "config.json"), ("CALIB_PATH", "calibration.json"),
                        ("ACCURACY_LOG", "accuracy_log.csv"), ("EVENTS_LOG", "events.csv")):
        monkeypatch.setattr(C, name, str(tmp_path / ("auto_" + fname)))
