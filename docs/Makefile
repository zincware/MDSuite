# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
APIBUILD	  = sphinx-apidoc

# Copy Notebooks over from the examples directory
define copy_notebooks
	mkdir source/_notebooks/
	cp ../examples/notebooks/* source/_notebooks/
endef

# Clean up after build
define clean_generated_files
	rm -r source/_notebooks/
endef

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
#	@$(APIBUILD) -f -o ./source/_modules ../
	$(call copy_notebooks)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	$(call clean_generated_files)
