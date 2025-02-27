from setuptools import setup, find_packages

setup(
    name="mimic_clustering",  # or whatever name you prefer
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'seaborn',
        'matplotlib',
        'google-auth'
        # add other dependencies as needed
    ]
)
