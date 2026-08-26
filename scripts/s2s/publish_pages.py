"""Assemble the gh-pages tree for the AfricaS2S Pages site.

Installs the shared site shell (hub index.html + theme.css) at the pages root and
rebuilds the s2s/ subtree from a freshly-rendered build dir, leaving sibling
subtrees (e.g. seasonal/) untouched.

Invocation:
  uv run python -m scripts.s2s.publish_pages \\
      --site-src site --s2s-build build/s2s --pages-root gh-pages-checkout
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_SHELL_FILES = ("index.html", "theme.css")


def publish_pages(*, site_src: Path, s2s_build: Path | None, pages_root: Path) -> None:
    """Assemble the gh-pages tree in ``pages_root`` (no git operations)."""
    site_src = Path(site_src)
    pages_root = Path(pages_root)
    pages_root.mkdir(parents=True, exist_ok=True)

    # 1. Install the shared shell (hub + theme) at the root.
    for fname in _SHELL_FILES:
        src = site_src / fname
        if not src.is_file():
            raise FileNotFoundError(f"site shell file missing: {src}")
        shutil.copy2(src, pages_root / fname)

    # 2. Rebuild only s2s/ from the fresh build, if it exists and is non-empty.
    #    Sibling subtrees (e.g. seasonal/) are never touched.
    if s2s_build is not None:
        s2s_build = Path(s2s_build)
        if s2s_build.is_dir() and any(s2s_build.iterdir()):
            dest = pages_root / "s2s"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(s2s_build, dest)
    # else / empty build: leave any existing pages_root/s2s intact.


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site-src", type=Path, default=Path("site"))
    ap.add_argument("--s2s-build", type=Path, default=None)
    ap.add_argument("--pages-root", required=True, type=Path)
    args = ap.parse_args()
    publish_pages(site_src=args.site_src, s2s_build=args.s2s_build, pages_root=args.pages_root)


if __name__ == "__main__":
    _cli()
