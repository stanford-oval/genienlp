# Reporting Issues
Did your `genienlp` command crash, or give you unexpected results? Open a new issue on GitHub. Please include the following items in your issue:

- What was the command that resulted in this bug? Ideally, include a short, self-contained code snippet that allows us to reproduce the bug.
- The full traceback if an exception is raised.
- Your operating system type and version, the versions of Python (using `python --version`), PyTorch (using `pip freeze | grep torch`) and transformers (using `pip freeze | grep transformers`) that you have installed.

# Contributing Code
We use the following tools to automatically check and enforce code style standards:

- [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks)
- [Pycln](https://github.com/hadialqattan/pycln): finds and removes unused Python import statements.
- [isort](https://github.com/PyCQA/isort): sorts Python imports alphabetically, and automatically separate them into sections and by type.
- [_Black_](https://github.com/psf/black) and [flake8](https://github.com/PyCQA/flake8): does code style checks for Python.

To automatically run these tools before each of your commits, follow these steps:

1. After cloning this repository, install `pre-commit`:

    ```
    pip install pre-commit
    ```

1. Run all pre-commit hooks (might take a minute the first time you run it):

    ```
    pre-commit run --all-files
    ```

Now you can commit the usual way (using `git commit -m "commit message"`).

Don't forget to write [good commit messages](https://chris.beams.io/posts/git-commit/).
