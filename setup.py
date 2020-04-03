from setuptools import setup, find_packages

setup(
    name="LambdaZero",
    version="0.1",
    packages=find_packages(include=["LambdaZero", "LambdaZero.*"]),
)

