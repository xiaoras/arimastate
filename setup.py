import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="arimastate",
    version="0.0.3",
    author="Andrea Dapor",
    author_email="andrea.dapor@gmail.com",
    description="python module for state space formulation of arima models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xiaoras/arimastate",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'sympy']
)