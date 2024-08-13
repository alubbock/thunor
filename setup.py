from setuptools import setup
import versioneer
import os


def main():
    this_directory = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_directory, 'README.md'), 'r') as f:
        long_description = f.read()

    setup(
        name='thunor',
        version=versioneer.get_version(),
        description='Dose response curve and drug induced proliferation '
                    '(DIP) rate fits and visualisation',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Alex Lubbock',
        author_email='code@alexlubbock.com',
        url='https://www.thunor.net',
        packages=['thunor', 'thunor.converters'],
        python_requires='>=3.10',
        install_requires=['numpy', 'scipy', 'pandas', 'plotly', 'seaborn',
                          'tables'],
        extras_require={
            'test': ['pytest', 'nbval', 'django', 'nbformat', 'flake8',
                     'codecov', 'pytest-cov'],
            'docs': ['sphinx', 'sphinx-rtd-theme', 'mock', 'nbsphinx',
                     'ipykernel'],
        },
        cmdclass=versioneer.get_cmdclass(),
        zip_safe=True,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
        ]
    )


if __name__ == '__main__':
    main()
