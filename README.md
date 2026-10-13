# Requirements to Run

To run this Python application, make sure that you have Python installed on your machine. Also make sure that it is added to your system's PATH such that the `python --version` command runs successfully on any working directory.

# How to Run

1. Open a terminal with `EmployeeAttritionSimulation` as the working directory
2. Run this command to install dependencies: `pip install -r requirements.txt`
    - **Fallback:** If the command results in an error, run this command instead:
        `pip install numpy pandas scipy scikit-learn matplotlib PyQt6`
3. Run the application: `python app.py`
    - If the application does not run at start-up or gives an error at runtime, then your machine's Python version may be insufficient to run the application. If any package/module issues occur even after installing the dependencies, try updating your Python version to 3.13.0 or 3.13.1.