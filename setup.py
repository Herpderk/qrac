from setuptools import setup, find_packages

setup(
    name='qrac',
    version='0.0.0',
    packages=find_packages(include=[
        'qrac', 'qrac.*'
    ]),
    install_requires=[
        'numpy',
        'scipy',
        'casadi',
        'matplotlib',
        'acados_template',
    ]
)