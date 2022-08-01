#!/usr/bin/env python
# coding: utf-8

# # Intro to Python Libraries 

# Python libraries give users access to modules that can perform certain functions without the user having to write their own code.

# ## Numpy

# https://numpy.org/
# 
# NumPy is the fundamental package for numerical computation. It defines the numerical array and matrix types and basic operations on them.
# 
# There is a wide range of mathematical functionality provided in the numpy library (See for example https://numpy.org/doc/stable/reference/routines.html)
# 
# Some of key submodules include:
# * numpy.linalg - linear algebra
# * numpy.fft - Fourier transforms
# * numpy.random - random sampling and various distributions

# ## SciPy 

# The “SciPy ecosystem” of scientific computing in Python builds upon a small core of packages: https://www.scipy.org/about.html

#  https://www.scipy.org/
# 
# * SciPy is the core package for scientific routines in Python 
# * The SciPy library is a collection of numerical algorithms and domain-specific toolboxes, including signal processing, optimization, statistics, and much more.
# * it operates efficiently on numpy arrays and the two are intended to work together
# * Many sub-modules are available:
#   * *scipy.cluster* - Vector quantization / Kmeans
#   * *scipy.constants* - Physical and mathematical constants
#   * *scipy.fftpack* - Fourier transform
#   * *scipy.integrate* - Integration routines
#   * *scipy.interpolate* - Interpolation
#   * *scipy.io* - Data input and output
#   * *scipy.linalg* - Linear algebra routines
#   * *scipy.ndimage* - n-dimensional image package
#   * *scipy.odr* - Orthogonal distance regression
#   * *scipy.optimize* - Optimization
#   * *scipy.signal* - Signal processing
#   * *scipy.sparse* - Sparse matrices
#   * *scipy.spatial* - Spatial data structures and algorithms
#   * *scipy.special* - Any special mathematical functions
#   * *scipy.stats* - Statistics

# ## Matplotlib

# https://matplotlib.org/
# 
# 
# * Matplotlib is a mature and popular plotting package that provides publication-quality 2-D plotting, as well as rudimentary 3-D plotting. 
# * "Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python."
# * "Matplotlib makes easy things easy and hard things possible."
# * Matplotlib was built on the NumPy and SciPy frameworks and initially made to enable interactive Matlab-like plotting via gnuplot from iPython
# * Gained early traction with support from the Space Telescope Institute and JPL
# * Easily one of the go-to libraries for academic publishing needs
#   * Create publication-ready graphics in a range of formats
#   * Powerful options to customize all aspects of a figure 
# * Matplotlib underlies the plotting capabilities of other libraries such as Pandas, Seaborn, and plotnine

# ## Pandas

# https://pandas.pydata.org/
# 
# Pandas data-manipulation capabilities are built on top of NumPy, utilizing its fast array processing, and its graphing capabilities are built on top of Matplotlib.
# 
# * "pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language."
# 
# * It may be one of the most widely used tools for data munging
#   * present data in nice formats
#   * multiple convenient methods for filtering data
#   * work with a variety of data formats (CSV, Excel, …)
#   * convenient functions for quickly plotting data
# 
# * Name comes from panel data, also play on python data analysis
# 
# Data manipulation in Python can be greatly facilitated with the Pandas library, and it may be one of the most widely used tools for data science.
# 
# Pandas data-manipulation capabilities are built on top of NumPy, utilizing its fast array processing, and its graphing capabilities are built on top of Matplotlib.
