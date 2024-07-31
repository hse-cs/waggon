from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="waggon",
    version="0.0.5",
    description="Wasserstein global gradient-free optimisation methods library.",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hse-cs/waggon",
    author="Tigran Ramazyan",
    author_email="ramazyant@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    install_requires=["matplotlib>=3.7.5", "numpy>=1.24.1", "scipy>=1.14.0",
                      "setuptools>=72.1.0", "torch>=2.2.2", "tqdm>=4.66.4"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.11",
)

