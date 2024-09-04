from setuptools import setup
import versioneer


def main():
    setup(
        name='thunor',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
    )


if __name__ == '__main__':
    main()
