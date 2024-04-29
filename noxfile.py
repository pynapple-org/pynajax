import os
import shutil
from pathlib import Path

import nox


def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

to_exclude = [
    "test_numpy_compatibility.py", # Pynajax has its own
    ]

#nox --no-venv -s linters
#nox --no-venv -s tests

@nox.session(name="linters")
def linters(session):
    """Run linters"""
    session.run("ruff", "check", "src", "--ignore", "D")

@nox.session(name="tests")
def tests(session):
    """Run the test suite."""

    nap_test_path = Path("../pynapple/tests")
    path = Path("tests/")

    # Copying test files
    tocopy = os.listdir(nap_test_path)
    tocopy = list(filter(lambda x: x[0:5] == "test_" and x not in to_exclude, tocopy))
    for f in tocopy:
        shutil.copy(nap_test_path.joinpath(f), path.joinpath(f))

    # Copying nwbfiletest and npzfiletest
    copy_and_overwrite(nap_test_path.joinpath("nwbfilestest"), path.joinpath("nwbfilestest"))

    try:
        session.run("pytest")#, "--pdb", "--pdbcls=IPython.terminal.debugger:Pdb")
    finally:
        # Remove files
        for f in tocopy:
            try:
                os.remove(path.joinpath(f))
            except OSError:
                pass
        # Remove folders
        shutil.rmtree(path.joinpath("nwbfilestest"))









