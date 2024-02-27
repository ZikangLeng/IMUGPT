depsOK = True
import numpy

try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
    if depsOK:
        setup(
            name = "imusim",
            version = "0.2",
            author = "Alex Young and Martin Ling",
            license = "GPLv3",
            url = "http://www.imusim.org/",
            install_requires = ["simpy>=2.3,<3", "pyparsing"],
            packages = find_packages(),
            include_dirs = [numpy.get_include()],
            ext_modules = [
                Extension("imusim.maths.quaternions",
                    ['imusim/maths/quaternions.c']),
                Extension("imusim.maths.quat_splines",
                    ['imusim/maths/quat_splines.c']),
                Extension("imusim.maths.vectors",['imusim/maths/vectors.c']),
                Extension("imusim.maths.natural_neighbour",[
                    'imusim/maths/natural_neighbour/utils.c',
                    'imusim/maths/natural_neighbour/delaunay.c',
                    'imusim/maths/natural_neighbour/natural.c',
                    'imusim/maths/natural_neighbour.c'])]
        )
except ImportError:
    print("Setuptools must be installed - see http://pypi.python.org/pypi/setuptools")