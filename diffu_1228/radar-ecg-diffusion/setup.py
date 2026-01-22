from setuptools import setup, find_packages

setup(
    name='radar-ecg-diffusion',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A diffusion model for reconstructing ECG data from millimeter-wave radar data.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.9.0',
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'pyyaml',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)