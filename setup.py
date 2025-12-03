from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="surfacepatcher",
    version="0.2.0",
    author="Alper Celik",
    author_email="alper.celik@sickkids.ca",
    description="Protein surface comparison using geometric, topological, and hybrid methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/celalp/surfacepatcher",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", #this is because of open3d
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
