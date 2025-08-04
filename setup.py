from setuptools import setup, find_packages

setup(
    name="information_leakage_detector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "fastapi",
        "uvicorn",
        "scikit-learn",
        "numpy",
        "pandas"
    ],
    python_requires=">=3.1",  # Specify Python version
)