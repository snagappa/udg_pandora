from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("rfs_merge", ["rfs_merge.pyx"])]

setup(
    name = "rfs_merge",
    cmdclass = {'build_ext': build_ext},
    extra_compile_args=["-O3", "-ffast-math"],
    ext_modules = ext_modules
    )
