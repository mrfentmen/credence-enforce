"""Shared test configuration.

Puts the repository root on `sys.path` so test modules can `import credence`
without each file repeating the same path manipulation above its imports. That
repetition is what produced 39 `E402 module-import-not-at-top-of-file` violations
across `tests/` — the import genuinely was not at the top, because it could not
be until the path existed.

Files that already contain their own `sys.path.insert` are unaffected: a second
insert of the same directory is a no-op.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
