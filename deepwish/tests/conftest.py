"""Shared pytest configuration for path setup and warning suppression."""

import os
import sys
import warnings


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

warnings.simplefilter("ignore", DeprecationWarning)


def pytest_configure(config):  # pragma: no cover - test-time configuration
    pass
