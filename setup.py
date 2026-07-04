import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

extensions = [
    Extension(
        name="waterTools.c_parser",  # il modulo risultante sarà waterTools.c_parser
        sources=["waterTools/c_parser.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="waterTools.c_parser_bis",  # il modulo risultante sarà waterTools.c_parser
        sources=["waterTools/c_parser_upos.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="waterTools",
    version="0.1.0",
    description="Tools per analisi spettrali e Wigner sampling dell'acqua",
    author="Alessandro Serra",
    packages=find_packages(),  # trova automaticamente waterTools
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
    install_requires=[
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "wigner-exc = waterTools.wignerEXC2B:main",
            "exyz2gpumd = waterTools.exyz2gpumd:main",
        ],
    },
)
