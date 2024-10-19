import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from setuptools import setup

setuptools.setup(
    name='maskmoment',
    version='1.2.2',
    description='Masked moments of radio data cubes',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Tony Wong',
    author_email = 'tonywong94@gmail.com',
    install_requires = ['numpy>=1.8.0',
                        'scipy',
                        'astropy',
                        'radio_beam',
                        'spectral_cube'],
    url='https://github.com/tonywong94/maskmoment',
    packages=['maskmoment'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
