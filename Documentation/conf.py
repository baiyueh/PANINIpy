project = 'PANINIpy: Packageof Algorithms for Nonparametric Inference on Networks in Python'
copyright = '2024, Baiyue'
author = 'Baiyue'

extensions = [
    'sphinx.ext.imgmath',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinxcontrib.bibtex'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
highlight_language = 'python'
bibtex_bibfiles = ['paper.bib']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

autosectionlabel_prefix_document = True
iimgmath_image_format = 'svg'  
imgmath_use_preview = True  
imgmath_add_tooltips = True  
imgmath_font_size = 12  
imgmath_latex = 'latex'
imgmath_latex_preamble = r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{bm}
'''
