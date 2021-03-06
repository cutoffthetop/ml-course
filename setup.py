from setuptools import setup, find_packages


setup(
    name='mlcourse',
    version='0.3.0.dev0',
    author='Nicolas Drebenstedt',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    license='Apache-2.0',
    namespace_packages=['mlcourse'],
    setup_requires=['setuptools_git'],
    install_requires=[
        'keras',
        'matplotlib',
        'graphviz',
        'numpy',
        'opencv-contrib-python',
        'pandas',
        'keras',
        'scikit-learn',
        'scipy',
        'setuptools',
        'sklearn',
        'tensorflow'
    ],
    entry_points={
        'console_scripts': [
            'contour=mlcourse.contour:main',
            'neural=mlcourse.neural:main',
            'digits=mlcourse.digits:main',
            'pca=mlcourse.contour:pca',
            'iris=mlcourse.iris:main',
            'wine=mlcourse.wine:main'
        ]
    }
)
