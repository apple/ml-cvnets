# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = cvnets
SOURCEDIR     = source
BUILDDIR      = _build
AUTOGENDIR    = $(SOURCEDIR)/autogen


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	echo '$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)'
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

rst:
	sphinx-apidoc -f -o $(AUTOGENDIR) ../data/
	sphinx-apidoc -f -o $(AUTOGENDIR) ../cvnets/
	sphinx-apidoc -f -o $(AUTOGENDIR) ../engine/
	sphinx-apidoc -f -o $(AUTOGENDIR) ../loss_fn/
	sphinx-apidoc -f -o $(AUTOGENDIR) ../loss_landscape/
	sphinx-apidoc -f -o $(AUTOGENDIR) ../optim/
	sphinx-apidoc -f -o $(AUTOGENDIR) ../metrics/
	sphinx-apidoc -f -o $(AUTOGENDIR) ../options/
	sphinx-apidoc -f -o $(AUTOGENDIR) ../utils/
	rm $(AUTOGENDIR)/modules.rst

github_pages:
	@make rst
	@make clean html
	@cp -a ./_build/html/ .
