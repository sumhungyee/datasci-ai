from setuptools import setup, find_packages

setup(
    name='your-library-name',
    version='1.0.0',
    author='Sum Hung Yee',
    author_email='hungyee2013@email.com',
    url='https://https://github.com/sumhungyee/datasci-ai/',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        'transformers',
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',
    ],
)