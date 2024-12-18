from pathlib import Path
from setuptools import setup, find_packages

def set_repository_path():
    """Set the repository path as an environment variable in .env."""
    from dotenv import set_key

    # Get the repository path
    repo_path = Path(__file__).resolve().parent

    # Create or update .env file
    env_file = repo_path / ".env"
    set_key(str(env_file), "REPO_PATH", str(repo_path))

    print(f"Repository path set in .env as REPO_PATH={repo_path}")

# Set the repository path
print("Setting repository path...")
set_repository_path()
print("Setup script executed!")

# Formal setuptools setup
setup(
    name="resum",
    version="1.0.0",
    description="A Python package for a rare event surrogate model (RESuM)",
    author="Ann-Kathrin Schuetz",
    author_email="annkaschue@gmail.com",
    packages=["resum"],#find_packages(where="."),
    #package_dir={"": "."},
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "pyyaml",
        "ipywidgets",
        "pandas",
        "tqdm",
        "dask[dataframe]",
        "python-dotenv",
        "imbalanced-learn==0.8.1",
        "termcolor",
        "scipy",
        "GPy",
        "emukit",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)