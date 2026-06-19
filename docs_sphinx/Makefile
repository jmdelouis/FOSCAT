# Minimal Sphinx Makefile

SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean livehtml install

help:
	@echo "Usage:"
	@echo "  make install    install doc dependencies"
	@echo "  make html       build HTML docs"
	@echo "  make clean      remove build artefacts"
	@echo "  make livehtml   auto-rebuild on file changes (requires sphinx-autobuild)"

install:
	pip install -r requirements.txt

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html
	@echo ""
	@echo "Build finished. Open $(BUILDDIR)/html/index.html"

clean:
	rm -rf $(BUILDDIR) autoapi

livehtml:
	sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html \
	    --ignore "*/autoapi/*" \
	    --open-browser
