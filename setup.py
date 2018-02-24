from setuptools import setup
import versioneer


def main():
    setup(name='thunor',
          version=versioneer.get_version(),
          description='Dose response curves and drug induced proliferation '
                      '(DIP) rates in Python',
          author='Alex Lubbock',
          author_email='code@alexlubbock.com',
          packages=['thunor'],
          install_requires=['numpy', 'scipy', 'pandas', 'plotly', 'seaborn'],
          tests_require=['pytest', 'nbformat', 'nbconvert'],
          cmdclass=versioneer.get_cmdclass(),
          zip_safe=True
    )


if __name__ == '__main__':
    main()
