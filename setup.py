from setuptools import find_packages, setup
import os


install_requires = [
    "tensorflow",
    "numpy",
    "pandas",
    "python-dotenv",
]

true_set = {'true','t', 'y', 'yes'}

is_developer = None

# def read_developer_acces():
#     if os.path.isfile(".env"):
#         for line in open(".env").read():
#             if line.startswith()


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