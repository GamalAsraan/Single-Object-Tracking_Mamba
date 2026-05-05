# TODO: Implement optional FastAPI/Flask server
import sys
import os

# Ensure we can import from app/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runtime.trackingmamba_runtime import TrackingMambaRuntime

def main():
    # TODO: Start inference API server
    pass

if __name__ == "__main__":
    main()
