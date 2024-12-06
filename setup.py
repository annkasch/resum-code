import os
import subprocess
from pathlib import Path

def install_packages():
    """Install necessary Python packages."""
    required_packages = [
        "numpy",
        "python-dotenv"
    ]
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", *required_packages])

def set_repository_path():
    """Set the repository path as an environment variable in .env."""
    from dotenv import set_key

    # Get the repository path
    repo_path = Path(__file__).resolve().parent

    # Create or update .env file
    env_file = repo_path / ".env"
    set_key(str(env_file), "RESUM_PATH", str(repo_path))

    print(f"Repository path set in .env as REPO_PATH={repo_path}")

if __name__ == "__main__":
    print("Installing required packages...")
    install_packages()
    print("Setting repository path...")
    set_repository_path()
    print("Setup complete!")