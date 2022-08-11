import subprocess
import sys

from ad_denoise import __version__


def test_cli_version_shows_version():
    cmd = [sys.executable, "-m", "ad_denoise", "--version"]
    assert __version__ == subprocess.check_output(cmd).decode().strip()


def test_cli_help_shows_help():
    cmd = [sys.executable, "-m", "ad_denoise", "--help"]
    assert (
        subprocess.check_output(cmd)
        .decode()
        .strip()
        .startswith("Usage: python -m ad_denoise")
    )


def test_cli_noargs_shows_help():
    cmd = [sys.executable, "-m", "ad_denoise"]
    assert (
        subprocess.check_output(cmd)
        .decode()
        .strip()
        .startswith("Usage: python -m ad_denoise")
    )
