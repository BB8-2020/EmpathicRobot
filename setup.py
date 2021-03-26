from setuptools import find_packages, setup

DEVELOPER = True

install_requires = []

if DEVELOPER:
    with open('requirements-dev.txt') as file:
        install_requires.extend(file.read().splitlines())

with open('requirements.txt') as file:
    install_requires.extend(file.read().splitlines())



setup(
    name='EmpathicRobot',
    version='1.0.0',
    description='EmpathicRobot opzet',
    author='Hogeschool Utrecht AI - 2021',
    author_email='maryadumak@gmail.com',
    packages=find_packages(),
    install_requires=install_requires,
  
)