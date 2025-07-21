# run.py
# This is the entry point for PyInstaller.

# By importing the 'ifi' package, we ensure that all submodules
# are correctly found, and relative imports within the package work.
from ifi import main

if __name__ == '__main__':
    # Run the main function from the ifi package's main module.
    main.main() 