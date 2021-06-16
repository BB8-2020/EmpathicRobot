---
description: >-
  This page describes the linting procedures set up in the "EmpaticRobot"
  repository.
---

# Linting

## GitHub actions \(CI/CD\)

The pull requests on `main`branches will be tested using a [Github Actions](https://github.com/features/actions) CI/CD pipeline. The tests are as follows:

* `pytest` checks if the unit-tests are successful
* `flake8` does the following linting checks
  * &lt; 140 chars per line
  * A default python check for syntax \(no undefined/non-compiling code etc\)
  * &lt; 6 [cyclomatic complexity](https://www.brandonsavage.net/code-complexity-and-clean-code/)

We do ignore some of the `flake8` rules. Check `setup.cfg` for more information. 



## Running tests and linting locally

To run the `flake8` locally, follow this:

```text
$ pip install -r requirements.txt
$ flake8 . --config="./setup.cfg"
```

To run the tests locally, you can use this:

```text
$ pytest . -vs 
```

Or choose to not run the test that take long to run by running this:

```text
$ pytest . -v -m "not long"
```

