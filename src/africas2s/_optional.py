def require_optional(package_name, install_hint=None):
    """Import and return a package, raising a clear error if missing."""
    try:
        return __import__(package_name)
    except ImportError:
        hint = install_hint or f"pip install {package_name}"
        raise ImportError(
            f"{package_name} is required for this feature. "
            f"Install with: {hint}"
        )
