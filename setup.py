from setuptools import setup
import versioneer


def main():
    setup(name='pydrc',
          version=versioneer.get_version(),
          description='Dose response curves and drug induced proliferation '
                      '(DIP) rates in Python ',
          author='Alex Lubbock',
          author_email='code@alexlubbock.com',
          packages=['pydrc'],
          install_requires=['numpy', 'scipy', 'pandas', 'plotly', 'seaborn'],
          cmdclass=versioneer.get_cmdclass()
    )

if __name__ == '__main__':
    main()
