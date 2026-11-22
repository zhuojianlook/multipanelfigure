"""/api/health echoes the launch token the app started this engine with, so
the app can tell its own engine from another program on the same port; the
token is not passed on to scripts the engine runs."""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROBE = """
import os
from fastapi.testclient import TestClient
import api_server
c = TestClient(api_server.app)
d = c.get('/api/health').json()
print(repr((d.get('status'), d.get('token'), os.environ.get('MPF_LAUNCH_TOKEN'))))
"""


def _probe(env_token):
    env = dict(os.environ)
    env.pop("MPF_LAUNCH_TOKEN", None)
    if env_token is not None:
        env["MPF_LAUNCH_TOKEN"] = env_token
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=HERE, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return eval(out.stdout.strip().splitlines()[-1])


def test_health_echoes_launch_token_and_hides_it_from_children():
    assert _probe("abc123") == ("ok", "abc123", None)


def test_health_without_token_reports_empty():
    assert _probe(None) == ("ok", "", None)
