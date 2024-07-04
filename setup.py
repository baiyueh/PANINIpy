# from distutils.core import setup
# import pathlib
# import setuptools


# setuptools.setup(
#     name='ScholarCodeCollective',
#     version='0.1.4',
#     description='A collective library for the code behind several academic papers',
#     long_description=open('README.md').read(),
#     long_description_content_type='text/markdown',
#     url='https://google.com',
#     author='Author Name', 
#     author_email='author_email@mail.com',
#     license='The Unlicense',
#     projects_urls={
#         "Documentation": "x",
#         "Source": "https://github.com"
#     },
#     python_requires=">3.9,<3.12",
#     packages=setuptools.find_packages(),
#     include_package_data=True,
#     entry_points={"console_scripts": ["paper = paper.cli:main"]},

# )

# extensions = [
#     Extension("ScholarCodeCollective.Community_Representatives_main.functions",
#               ["ScholarCodeCollective/Community_Representatives_main/functions.pyx"],
#               language='c++')  
# ]
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# extensions = [
#     Extension("ScholarCodeCollective.Community_Representatives_main.functions",
#               ["ScholarCodeCollective/Community_Representatives_main/functions.pyx"],
#               include_dirs=[numpy.get_include()],
#               language='c++')  
# ]

setup(
    name='ScholarCodeCollective',
    version='0.1.8.3',
    description='A collective library for the code behind several academic papers',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://google.com',
    author='Author Name',
    author_email='author_email@mail.com',
    license='The Unlicense',
    projects_urls={
        "Documentation": "x",
        "Source": "file:///D:/Research%20HKU/PYPI_lib/Documentation/_build/html/index.html"
    },
    python_requires=">=3.9, <3.12",
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["paper = paper.cli:main"]},
    #ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
