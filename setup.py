from setuptools import setup, find_packages, Extension

setup(
    name='paninipy',
    version='0.6',
    description='Packageof Algorithms for Nonparametric Inference on Networks in Python',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://paninipy.readthedocs.io/en/latest/index.html',
    author='Baiyue He',
    author_email='baiyueh@hku.hk',
    license='The Unlicense',
    projects_urls={
        "Documentation": "https://paninipy.readthedocs.io/en/latest/index.html",
        "Source": "https://paninipy.readthedocs.io/en/latest/index.html"
    },
    python_requires=">=3.9, <3.12",
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["paper = paper.cli:main"]},
)
