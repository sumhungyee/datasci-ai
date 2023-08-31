from setuptools import setup, find_packages

setup(
    name='datasci_ai',
    version='1.1.0',
    author='Sum Hung Yee',
    author_email='hungyee2013@email.com',
    url='https://https://github.com/sumhungyee/datasci-ai/',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 2 - Planning',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        'ctransformers',
        'transformers',
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'plotly',
        'matplotlib'
    ],
)