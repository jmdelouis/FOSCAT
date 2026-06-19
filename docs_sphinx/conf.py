# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "FOSCAT"
year = "2026"
author = "Jean-Marc Delouis, Theo Foulquier"
copyright = f"{year}, {author}"
release = "2026.04.1"

root_doc = "index"
master_doc = "index"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "myst_nb",
]

# ---------------------------------------------------------------------------
# MyST
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "deflist",
]

nb_execution_mode = "off"

# ---------------------------------------------------------------------------
# AutoAPI
# ---------------------------------------------------------------------------
autoapi_dirs = ["../src/foscat"]
autoapi_type = "python"
autoapi_output_dir = "autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_keep_files = True
autoapi_python_use_implicit_namespaces = True

# ---------------------------------------------------------------------------
# Suppress warnings
# ---------------------------------------------------------------------------
suppress_warnings = [
    "autoapi.python_import_resolution",
    "autoapi",
    "myst.header",
    "ref.python",
    "intersphinx.external",
]

# ---------------------------------------------------------------------------
# Napoleon (NumPy-style docstrings)
# ---------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "healpy": ("https://healpy.readthedocs.io/en/latest/", None),
}

# ---------------------------------------------------------------------------
# HTML — PyData theme
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "FOSCAT"
html_theme_options = {
    "navigation_depth": 4,
    "show_toc_level": 2,
    "github_url": "https://github.com/jmdelouis/FOSCAT",
    "icon_links_label": "Quick Links",
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
}

# ---------------------------------------------------------------------------
# Source suffixes
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
