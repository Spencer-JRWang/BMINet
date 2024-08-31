import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi (if you dont need delete it)
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# you need to change all these
VERSION = '0.0.10'
DESCRIPTION = 'Machine Learning and Graph based tool for detecting and analyzing Bone-Muscle Interactions'
LONG_DESCRIPTION = 'Machine Learning and Graph based tool for detecting and analyzing Bone-Muscle Interactions'

from setuptools import setup, find_packages

setup(
    name="BMINet",
    version=VERSION,
    author="Spencer Wang",
    author_email="jrwangspencer@stu.suda.edu.cn",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.0',
        'pandas',
        'scikit-learn',
        'networkx',
        'matplotlib',
        'scipy',
        'shap',
        'seaborn',
        'xgboost',
        'lightgbm',
        'catboost',
        'community',
        'tqdm'
    ],
    include_package_data=True,
    package_data={
        'BMINet': ['Interaction/*.pkl'],
    },
    keywords=['python', 'BMINet', 'Interaction', 'Network', 'Bone-Muscle', 'windows', 'mac', 'linux'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='>=3.8',
)

