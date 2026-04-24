# FOSCAT Sphinx entry point

This is a minimal Sphinx bridge. The main documentation source is the MkDocs `docs/` directory.

To build with Sphinx, either copy the Markdown pages from `../docs` into this directory or configure your build system to include them.

```bash
pip install sphinx myst-parser
sphinx-build -b html docs_sphinx _build/html
```
