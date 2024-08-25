from setuptools import setup, find_packages
setup(
    name='NetBMI',
    version='1.0',
    description='Machine learning and network based tool for detecting Bone-Muscle interactions',
    author='Wang Jingran',
    author_email='jrwangspencer@stu.suda.edu.cn',
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/Spencer-JRWang/NetBMI',
    install_requires=[
        "pandas",
        "matplotlib",
        "numpy",
        "rpy2",
        "seaborn",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "xgboost",
        "shap",
        "tqdm"
    ],
    python_requires='>=3.8',
)
