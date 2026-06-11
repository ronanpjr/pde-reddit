from setuptools import setup, find_packages

setup(
    name="pde_reddit_features",
    version="0.0.1",
    description="Feature extraction helpers for PDE Reddit project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
