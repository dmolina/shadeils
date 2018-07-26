from setuptools import setup, Extension
from setuptools.command.test import test as TestCommand
from Cython.Distutils import build_ext

import sys

cython = Extension('ea.cbenchmarks',
    sources = ['ea/cbenchmarks.pyx'],
#    include_dirs = ['include/']
)

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(self.pytest_args)
        sys.exit(errno)



setup(
	name='ea',
        version='0.1',
        author='Daniel Molina',
        author_email='daniel.molina@uca.es',
        description='Package for new DEs in development',
        long_description=open('README.txt').read(),
        license='GPL V3',
        url='https://github.com/dmolina/cec2013lsgo',
        packages=['ea'],
        install_requires=['cython', 'numpy'],
        ext_modules=[cython],
#	ext_modules=cythonize('benchmarks.pyx', annotated=True),
        cmdclass={'build_ext': build_ext, 'test': PyTest},
        classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    ]

)
