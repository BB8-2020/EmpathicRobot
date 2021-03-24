from setuptools import setup

install_requires = [
    'matplotlib',
    'seaborn',
    'numpy',
    'pandas',
]

setup(
    name='EmpathicRobot',
    version='1.0.0',
    description='EmpathicRobot opzet',
    author='Hogeschool Utrecht AI - 2021',
    author_email='maryadumak@gmail.com',
    packages=[],
    install_requires=install_requires,
    dependency_links=[
        # "git+https://github.com/BB8-2020/<repo_name>#egg=<repo_name>",
    ],
)
