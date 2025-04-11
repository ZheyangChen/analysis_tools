from setuptools import setup, find_packages

setup(
    name="analysis_tools",
    version="0.1.0",
    description="A collection of analysis functions for plotting and evaluation.",
    author="Zheyang Chen",
    author_email="zheyang.chen@stonybrook.edu",
    packages=find_packages(),  # This finds all packages (directories with __init__.py)
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "pyyaml"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)