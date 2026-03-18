import sys
import os

# Add the parent directory to sys.path so 'ydvt' package can be imported directly 
# when calling `python3 ydvt`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ydvt.main import main

if __name__ == "__main__":
    main()
