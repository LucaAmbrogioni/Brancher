from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='brancher',
      version='0.1.0',
      description='Write description',
      author='MindCodec',
      author_email='something@mindcodec.com',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/LucaAmbrogioni/Brancher",
      packages=find_packages(),
      install_requires=['tqdm>=4.31.1'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      )

__author__ = 'MindCodec'
