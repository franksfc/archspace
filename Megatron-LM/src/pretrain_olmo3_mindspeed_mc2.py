#!/usr/bin/env python3
"""OLMo3 MindSpeed entry point with the Ascend MC2 TP kernel enabled.

The flag must be present before importing ``pretrain_olmo3_mindspeed`` because
MindSpeed selects and installs its tensor-parallel linear patches during that
module's import-time runtime initialization.  Keeping this as a separate entry
point leaves every frozen non-MC2 launcher and checkpoint path unchanged.
"""

from __future__ import annotations

import sys


MC2_FLAG = "--use-ascend-mc2"


def main() -> None:
    if MC2_FLAG not in sys.argv:
        sys.argv.append(MC2_FLAG)

    from pretrain_olmo3_mindspeed import main as pretrain_main

    pretrain_main()


if __name__ == "__main__":
    main()
