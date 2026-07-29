"""A peer that dials the worker channel but cannot prove it is the worker.

Started as the "interpreter" an external launcher runs, so what gets exercised
is the launcher's own handshake. It stays alive after failing, the way a
stranger squatting on a loopback port would, leaving the launcher to decide on
its own that the worker it wanted is never arriving.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from multiprocessing.connection import Client

payload = json.loads(sys.stdin.readline())
address = payload["address"]
with contextlib.suppress(BaseException):
    Client(tuple(address) if isinstance(address, list) else address, authkey=b"not-the-key")
time.sleep(30)
