"""Sphinx configuration file for the Kinematic Arbiter documentation."""

import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "arbiter"
copyright = "2025, Spencer Maughan"
author = "Spencer Maughan"
release = "0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("."))

extensions = [
    "breathe",
    "exhale",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Breathe and Exhale configuration ----------------------------------------
# Setup for using Doxygen output with Sphinx

breathe_projects = {"KinematicArbiter": "../doxygen/xml"}
breathe_default_project = "KinematicArbiter"

exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C++ API Reference",
    "doxygenStripFromPath": "../..",
    "createTreeView": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2,
}
html_static_path = ["_static"]
