from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hideseek",
    version="1.0.0",
    author="HideSeek Development Team",
    author_email="99cvteam@gmail.com",
    description="A professional Python application for testing and evaluating camouflage effectiveness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/hideseek",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "hideseek=hideseek.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hideseek": ["data/*", "config.yaml"],
    },
)