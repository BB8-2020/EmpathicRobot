from setuptools import find_packages, setup
import os
import sys


install_requires = [
    "tensorflow",
    "numpy",
    "pandas",
]

if sys.platform.startswith('linux') or sys.platform == 'darwin':
    install_requires.extend(['pycocotools'])
elif sys.platform == 'win32':
    install_requires.extend(['pycocotools-windows'])
else:
    raise OSError(f"{sys.platform} is not supported")

true_set = {'true', '1', 't', 'y', 'yes'}

is_developer = None

if os.path.isfile('.env'):
    for line in open('.env').read().split('/n'):
        if line.replace(' ', '').startswith('DEVELOPER='):
            is_developer = ''.join(line.split('=')[1:]).lower() in true_set

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
        'pytest'
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