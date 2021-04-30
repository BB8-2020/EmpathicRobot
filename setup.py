from setuptools import find_packages, setup
import os

install_requires = [
    "tensorflow",
    "numpy",
    "pandas",
]
true_set = {'true', 't', 'y', 'yes'}


def read_developer_access() -> bool:
    """Reads if the developer access true else return a None."""
    developer = None
    if os.path.isfile(".env"):
        for line in open(".env").read().split('/n'):
            if line.replace(' ', '').startswith('DEVELOPER='):
                developer = ''.join(line.split('=')[1:]).lower() in true_set
    return developer


# Check if the user is developer
is_developer = read_developer_access()
if is_developer is None:
    is_developer = os.environ.get('DEVELOPER', 'false').lower() in true_set

# Install requirements if the user is developer
if is_developer:
    install_requires.extend([
        'flake8',
        'flake8-import-order',
        'flake8-blind-except',
        'flake8-builtins',
        'flake8-docstrings',
        'flake8-rst-docstrings',
        'flake8-logging-format',
        'mypy',
        'pytest',
    ])

setup(
    name='EmpathicRobot',
    version='1.1.0',
    description='Recognize emotions using camera images of the robot',
    author='Hogeschool Utrecht AI - 2021',
    author_email='maria.dukmak@student.hu.nl',
    packages=find_packages(),
    install_requires=install_requires,
)
