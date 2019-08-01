# Sample datasets

For open-modex four small sample datasets are translated into tabular datapackages, that can be read by oemof.tabular.

### Requirements

To run the script, make sure to install the requirements e.g. via pip in a virtual environment

    pip install -r requirements.txt

The gurobi-solver library has to be available too. You can change the solver in compute.py .

### Compute

To compute the datapackages run:

    python compute.py

This will create a results directory with all results and open a html page with the meta results.
