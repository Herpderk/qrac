from setuptools import setup, find_packages

setup(
    name='qrac',
    version='0.0.0',
    packages=find_packages(include=[
        'qrac', 'qrac.*'
    ]),
    install_requires=[
        'numpy==1.26.4',
        'scipy==1.12.0',
        'matplotlib==3.8.3',
        'casadi==3.6.3',
        'proxsuite==0.6.2',
        'cflib==0.1.25.1',
    ]
)
