import os
import codecs
import setuptools

# # Load long description
# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

with open('requirements.txt') as f:
    default_requirements = [
        line.strip() for line in f if line and not line.startswith('#')
    ]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name='cytoself',
    version=get_version("cytoself/__init__.py"),  # cytoself.__version__,  #
    author='',
    author_email='',
    description='An image feature extractor with self-supervised learning',
    url='https://github.com/royerlab/cytoself',
    project_urls={
        "Bug Tracker": 'https://github.com/royerlab/cytoself/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
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
    package_dir={"": f".{os.path.sep}"},
    packages=setuptools.find_packages(),
    install_requires=default_requirements,
    python_requires=">=3.6, <3.8",
)
