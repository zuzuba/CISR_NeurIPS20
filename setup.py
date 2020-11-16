import setuptools, platform

with open("README.md", "r", encoding='utf_8') as fh:
    long_description = fh.read()

install_requires=[
    'stable_baselines>=2.8.0,<2.9.0',
    'tensorflow>=1.15,<2.0.0',
    'numpy>=1.16.0,<1.19.0',
    'Box2D', 'GPyOpt', 'tabulate', 'matplotlib', 'scipy', 'GPy', 'gym',
    'cloudpickle', 'tabulate'
]

setuptools.setup(
    name='cisr',
    version='0.1.0',
    description='Implementation of the Curriculum Induction for Safe '
                'Reinforcement learning (CISR) framework',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/zuzuba/SafeCL',
    author='Matteo Turchetta',
    author_email='matteo.turchetta@inf.ethz.ch',
    keywords='safe reinforcement learning',
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=install_requires
)