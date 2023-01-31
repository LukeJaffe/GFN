# Global imports
import setuptools

# package name: osr = Object Search Research
PACKAGE = 'osr'

# Setup function
setuptools.setup(
    name='{}-lib'.format(PACKAGE),
    namespace_packages=[PACKAGE],
    version=open('VERSION').read().strip(),
    description='Object Search Research Library',
    packages=['osr', 'osr.data', 'osr.models', 'osr.engine', 'osr.losses',
        'osr.app', 'osr.viz'],
    package_dir={'osr': 'src'},
    entry_points={
        'console_scripts': [
            (
                '{pkg}_prep_cuhk = '
                '{pkg}.data.cuhk_utils:main'
                .format(pkg=PACKAGE)
            ),
            (
                '{pkg}_prep_prw = '
                '{pkg}.data.prw_utils:main'
                .format(pkg=PACKAGE)
            ),
            (
                '{pkg}_run = '
                '{pkg}.engine.main:main'
                .format(pkg=PACKAGE)
            ),
            (
                '{pkg}_search = '
                '{pkg}.app.search:main'
                .format(pkg=PACKAGE)
            ),
            (
                '{pkg}_model_convert = '
                '{pkg}.app.convert:main'
                .format(pkg=PACKAGE)
            ),
            (
                '{pkg}_model_shrink = '
                '{pkg}.app.shrink:main'
                .format(pkg=PACKAGE)
            ),
        ]
    }
) 
