from setuptools import setup, find_packages

setup(
    name='llambda',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Add any console scripts or entry points here
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        # Add more classifiers as needed
    ],
)
