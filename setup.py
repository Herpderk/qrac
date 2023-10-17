from setuptools import setup, find_packages

setup(
    name='aCBF_QP_MPC',
    version='0.0.0',
    packages=find_packages(include=[
        'aCBF_QP_MPC', 'aCBF_QP_MPC.*'
    ]),
    install_requires=[
        'numpy',
        'scipy',
        'casadi',
        'matplotlib',
        'acados_template',
    ]
)