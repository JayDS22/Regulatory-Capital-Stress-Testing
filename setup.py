from setuptools import setup, find_packages

setup(
    name="regulatory-capital-stress-testing",
    version="1.0.0",
    author="Jay Guwalani",
    author_email="jayguwalani@example.com",
    description="Banking regulatory compliance platform for automated stress testing and capital adequacy analysis under Basel III/IV frameworks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JayDS22/Regulatory-Capital-Stress-Testing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "jinja2>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "stress-test=cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial",
    ],
)
