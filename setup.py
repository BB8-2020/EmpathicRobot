from setuptools import find_packages, setup
import os
import sys
from dotenv import load_dotenv


install_requires = [
    "tensorflow",
    "numpy",
    "pandas",
    'python-dotenv',
]

if sys.platform.startswith('linux') or sys.platform == 'darwin':
    install_requires.extend(['pycocotools'])
elif sys.platform == 'win32':
    install_requires.extend(['pycocotools-windows'])
else:
    raise OSError(f"{sys.platform} is not supported")

true_set = {'true', '1', 't', 'y', 'yes'}

is_developer = None

load_dotenv()  # take environment variables from .env.

DEVELOP_MODE = os.getenv("DEVELOPER", "False").lower() in true_set # Read from DEVELOPER env variable elsewise use false.

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
        'python-dotenv',
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