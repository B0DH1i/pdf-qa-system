#!/usr/bin/env python
"""
PDF QA System API Launcher
This script starts the web interface for the PDF QA System.
"""

import os
import sys
from loguru import logger

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.interface.web.api import API

def main():
    """Initialize and start the API."""
    try:
        # Setup logging
        logger.info("PDF QA System starting...")
        
        # Create API instance
        port = int(os.environ.get("PORT", 8000))
        api = API(port=port)
        
        # Start the API
        host = "0.0.0.0"
        logger.info(f"Starting API on http://127.0.0.1:{port}")
        api.start(host=host, port=port)
    except Exception as e:
        logger.error(f"Error starting API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 