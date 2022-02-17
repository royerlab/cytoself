from os import path
import codecs
import setuptools

# Load long description
with open(
    path.join(path.abspath(path.dirname(__file__)), "README.md"), "r", encoding="utf-8"
) as fh:
    long_description = fh.read()

with open(path.join(path.abspath(path.dirname(__file__)), "requirements.txt")) as f:
    default_requirements = [
        line.strip() for line in f if line and not line.startswith("#")
    ]


def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="cytoself",
    version=get_version("cytoself/__init__.py"),
    author="",
    author_email="",
    license="BSD 3-Clause",
    description="An image feature extractor with self-supervised learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/royerlab/cytoself",
    project_urls={
        "Bug Tracker": "https://github.com/royerlab/cytoself/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    package_dir={"": f".{path.sep}"},
    packages=setuptools.find_packages(),
    install_requires=default_requirements,
    python_requires=">=3.7, <3.8",
)
