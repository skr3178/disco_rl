from setuptools import setup
import pathlib

# Read the long description from README
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="mathisfun-sokoban",
    version="1.0.0",
    py_modules=["mathisfun_sokoban", "__init__"],
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.20.0",
        "pygame>=2.0.0",
        "Pillow>=9.0.0",
    ],
    entry_points={
        "gymnasium.envs": [
            "MathIsFunSokoban-v0=mathisfun_sokoban:MathIsFunSokoban",
        ],
    },
    python_requires=">=3.8",
    description="A Gymnasium environment featuring all 60 Sokoban levels from MathsIsFun.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MathIsFun Sokoban Contributors",
    author_email="",
    url="https://github.com/yourusername/math_is_fun",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/math_is_fun/issues",
        "Documentation": "https://github.com/yourusername/math_is_fun/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/math_is_fun",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Puzzle Games",
    ],
    keywords="reinforcement-learning gymnasium sokoban puzzle rl environment",
    license="MIT",
    include_package_data=True,
    package_data={
        "": ["*.json"],
    },
)

