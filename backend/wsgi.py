"""
WSGI entry point for production deployment
Use with gunicorn or uwsgi
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Set environment
os.environ.setdefault('FLASK_ENV', 'production')

# Import app directly (we're in backend/ directory)
from app import app

if __name__ == "__main__":
    app.run()
