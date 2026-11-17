import os
import tempfile

os.environ["KS_WORKSPACE"] = tempfile.mkdtemp(prefix="ks_test_ws_")
