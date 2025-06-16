from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="TMG",
    version="1.0.0",
    author="dredFISH Development Team",
    description="TissueMultiGraph Analysis and Visualization Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas", 
        "scipy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "igraph",
        "pynndescent",
        "anndata",
        "umap-learn",
        "torch",
        "tqdm", 
        "colorcet",
        "xycmap",
        "colormath",

    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "jupyter",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 