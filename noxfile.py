import nox
import shutil
import os
from pathlib import Path

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

to_exclude = [
    "test_nwb.py",
    "test_numpy_compatibility.py",
    "test_neurosuite.py",
    "test_time_series.py",
    "test_tuning_curves.py",
    "test_spike_trigger_average.py",
    "test_abstract_tsd.py",
    ]

#nox --no-venv -s linters
#nox --no-venv -s tests

# @nox.session(name="linters")
# def linters(session):
#     """Run linters"""
#     session.run("ruff", "check", "src", "--ignore", "D")

@nox.session(name="tests")
def tests2(session):
    """Run the test suite."""
    
    nap_test_path = Path.home().joinpath("pynapple/tests")
    path = Path.home().joinpath("pynajax/tests")
    
    # Copying test files
    tocopy = os.listdir("../pynapple/tests/")
    tocopy = list(filter(lambda x: x[0:5] == "test_" and x not in to_exclude, tocopy))
    for f in tocopy:
        shutil.copy(nap_test_path.joinpath(f), path.joinpath(f))

    # Copying nwbfiletest and npzfiletest
    copy_and_overwrite(nap_test_path.joinpath("nwbfilestest"), path.joinpath("nwbfilestest"))
    copy_and_overwrite(nap_test_path.joinpath("npzfilestest"), path.joinpath("npzfilestest"))

    try:
        session.run("pytest")        
    finally:
        # Remove files
        for f in tocopy:
            try:
                os.remove(path.joinpath(f))
            except OSError:
                pass
        # Remove folders
        shutil.rmtree(path.joinpath("nwbfilestest"))
        shutil.rmtree(path.joinpath("npzfilestest"))
            

    






