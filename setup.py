import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi (if you dont need delete it)
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# you need to change all these
VERSION = '0.0.1'
DESCRIPTION = 'Machine Learning and Graph based tool for detecting and analyzing Bone-Muscle Interactions'
LONG_DESCRIPTION = 'Machine Learning and Graph based tool for detecting and analyzing Bone-Muscle Interactions'

setup(
    name="Net-BMI",
    version=VERSION,
    author="Spencer Wang",
    author_email="jrwangspencer@stu.suda.edu.cn",
    description=DESCRIPTION,
    # long_description_content_type="text/markdown",
    # long_description=long_description,
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'Net-BMI', 'Interaction','Network','Bone-Muscle','windows'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
