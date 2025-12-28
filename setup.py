from setuptools import setup, find_packages

setup(
    name="cxr-processing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
        'scikit-image>=0.18.0',
        'scipy>=1.7.0',
        'opencv-python>=4.5.0',
    ]
)