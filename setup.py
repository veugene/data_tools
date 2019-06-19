import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data_tools",
    version="1.0",
    author="Eugene Vorontsov",
    author_email="eugene.vorontsov@gmail.com",
    description="High performance data loading, preprocessing, or "
                "preparation for deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/veugene/data_tools",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'SimpleITK',
        'h5py',
        'bcolz'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
 
