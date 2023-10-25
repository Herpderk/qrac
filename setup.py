from setuptools import setup, find_packages

setup(
    name='ractoolbox',
    version='0.0.0',
    packages=find_packages(include=[
        'ractoolbox', 'ractoolbox.*'
    ]),
    install_requires=[
        'numpy',
        'scipy',
        'casadi',
        'matplotlib',
        'acados_template',
    ]
)