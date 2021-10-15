from setuptools import setup

setup(name='maskmoment',
    version='1.1',
    description='Masked moments of radio data cubes',
    author='Tony Wong',
    author_email = 'tonywong94@gmail.com',
    install_requires = ['numpy>=1.8.0',
                        'scipy',
                        'astropy',
                        'radio_beam',
                        'spectral_cube'],
    url='https://github.com/tonywong94/maskmoment',
    packages=['maskmoment'],
    )
