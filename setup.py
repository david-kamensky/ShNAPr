import os
import codecs

from setuptools import setup

base_dir = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()

setup(
    name='ShNAPr',
    version='2019.1',
    packages=['ShNAPr'],
    url='https://github.com/david-kamensky/ShNAPr',
    license='GNU LGPLv3',
    author='D. Kamensky',
    author_email='',
    description="Kirchhoff-Love shell forumulation for tIGAr/FEniCS",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
