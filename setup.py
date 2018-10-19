import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hierarpy",
    version="0.0.1",
    author="Lucas G. Goldstone",
    author_email="lu.goldstone@gmail.com",
    description="A python package for calculating animal dominance hierarchies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sacul-git/hierarpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'networkx',
    ],
)
