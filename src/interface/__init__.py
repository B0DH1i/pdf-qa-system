"""
User interface module.
Contains web and graphical interfaces.
"""

from .web import API
from .gui import run_gui, PDFQA

__all__ = [
    'API',      # Web API
    'run_gui',  # Start GUI application
    'PDFQA'     # PDF question-answer interface
] 