from setuptools import setup, Extension

module = Extension('_fast_sumtree',
                   sources = ['_fast_sumtree.c'])

setup(name = 'fast_sumtree',
      version = '1.0',
      description = 'A faster SumTree implementation',
      ext_modules = [module])
