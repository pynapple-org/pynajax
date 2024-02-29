import nox


@nox.session(name="tests")
def tests(session):
    """Run the test suite."""
    session.run("pytest")


@nox.session(name="linters")
def linters(session):
    """Run linters"""
    session.run("ruff", "check", "src", "--ignore", "D")

