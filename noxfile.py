import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ("lint", "test")

SRC = "mindmap"


def format_with_args(session: nox.Session, *args):
    session.run("autoflake", *args)
    session.run("isort", *args)
    session.run("black", *args)


MIN_PYTHON_VERSION = "3.11"


@nox.session(python=[MIN_PYTHON_VERSION])
def lint(session: nox.Session):
    """Runs linters and fixers"""
    try:
        session.run("poetry", "install", external=True)
        session.run("poetry", "check", "--lock", external=True)
        session.run("pyright", SRC)
        session.run("flake8", SRC)
        format_with_args(session, SRC, "--check")
    except Exception:
        session.error(
            "linting has failed. Run 'make format' to fix formatting and fix other errors manually"
        )


@nox.session(python=[MIN_PYTHON_VERSION])
def format(session: nox.Session):
    """Runs linters and fixers"""
    session.run("poetry", "install", external=True)
    format_with_args(session, SRC)


@nox.session(python=[MIN_PYTHON_VERSION])
def test(session: nox.Session):
    """Runs unit tests"""
    session.run("poetry", "install", external=True)
    session.run("pytest", "tests/unit_tests/")


@nox.session(python=[MIN_PYTHON_VERSION])
def update_graph(session: nox.Session):
    """Runs unit tests"""
    session.run("poetry", "install", "--with=update_data", external=True)
    session.run("python", "./update_data/update_graph.py")
