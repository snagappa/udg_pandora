from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("offline_slam_worker2", ["offline_slam_worker2.pyx"])]

setup(
    name = "offline_slam_worker2",
    cmdclass = {'build_ext': build_ext},
    extra_compile_args=["-O3", "-ffast-math"],
    ext_modules = ext_modules
    )
