import os
import tempfile

# Keep tests away from real covers and the real key before studio.config reads them.
os.environ.update(DATA_DIR=tempfile.mkdtemp(prefix="studio-test-"), NEBIUS_API_KEY="", BUDGET_USD="0.15")
