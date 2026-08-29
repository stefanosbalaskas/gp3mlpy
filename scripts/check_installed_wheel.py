"""Smoke-check an installed gp3mlpy distribution against the frozen API contract."""
from __future__ import annotations

import gp3mlpy


def main() -> None:
    exports = gp3mlpy.gp3ml_api_contracts().exports
    assert len(exports) == 127
    assert int((exports["stability"] == "stable").sum()) == 71
    assert int((exports["stability"] == "experimental").sum()) == 56
    assert bool(exports["present"].all())
    assert gp3mlpy.r_reference_version == "0.3.0"
    print("installed-wheel frozen-API smoke: PASS")


if __name__ == "__main__":
    main()
