from setuptools import find_packages, setup

setup(name="seqdata", packages=find_packages())

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="seqdata",
    version="0.0.1",
    author="Adam Klie",
    author_email="aklie@eng.ucsd.edu",
    description="Annotated sequence data",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/adamklie/SeqData",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
