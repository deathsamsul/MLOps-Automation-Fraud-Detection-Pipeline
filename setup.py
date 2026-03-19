from setuptools import setup, find_packages

setup(
    name="fraud_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "catboost",
        "mlflow",
        "evidently",
        "plotly",
        "requests",
        'pytest'
    ],
)