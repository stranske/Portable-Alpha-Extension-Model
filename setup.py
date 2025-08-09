from setuptools import find_packages, setup

setup(
    name="portable-alpha-extension-model",
    version="0.1.0",
    description="Portable alpha plus active extension model",
    author="Your Name",
    packages=find_packages(include=["pa_core*", "archive*"]),
    install_requires=[
        "numpy",
        "pandas",
        "openpyxl",
        "pydantic",
        "rich",
        "plotly>=5.19",
        "kaleido",
        "streamlit>=1.35",
        "python-pptx",
        "xlsxwriter",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": ["pa=pa_core.pa:main"],
    },
)
