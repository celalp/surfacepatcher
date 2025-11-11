from setuptools import setup, find_packages

setup(
    name='surfapatcher',
    version='0.1.0',
    description="create geodesic protein surface patches",
    author='Alper Celik',
    author_email='alper.celik@sickkids.ca',
    packages=find_packages(),
    zip_safe=False,
    package_data={"": ["*.json"]},
    include_package_data=True
)