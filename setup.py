from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='ScholarCodeCollective',
    version='0.1.8.7',
    description='A collective library for the code behind several academic papers',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://scholarcodecollective.readthedocs.io/en/latest/index.html',
    author='Baiyue He',
    author_email='baiyueh@hku.hk',
    license='The Unlicense',
    projects_urls={
        "Documentation": "https://scholarcodecollective.readthedocs.io/en/latest/index.html",
        "Source": "https://scholarcodecollective.readthedocs.io/en/latest/index.html"
    },
    python_requires=">=3.9, <3.12",
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["paper = paper.cli:main"]},
    #ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
