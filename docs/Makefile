# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

TUT_DST_DIR := tutorials

NOTEBOOKS := \
	fitting_procedure.ipynb \
	prior_sensitivity.rst \
	model_framework.rst \
	mstable_to_uvtable.rst \
	fit_convergence.rst \
	AS209_clean_profile.txt \
	AS209_continuum.npz \


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	#rm -rf $(TUT_DST_DIR)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile $(NOTEBOOK_TARGETS)

NOTEBOOK_TARGETS := $(patsubst %, $(TUT_DST_DIR)/%, $(NOTEBOOKS))


#$(TUT_DST_DIR)/%: $(TUT_SRC_DIR)/% $(TUT_DST_DIR)
#cp $< $@

.PHONY: help Makefile

$(TUT_DST_DIR):
	mkdir -p $(TUT_DST_DIR)


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html: Makefile $(NOTEBOOK_TARGETS)
	@echo $(NOTEBOOK_TARGETS)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
