import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import venv

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows VBS source launcher")
def test_source_launcher_uses_own_directory_and_local_python_from_unrelated_cwd(tmp_path):
    project = tmp_path / "project & local build"
    scripts = project / ".venv" / "Scripts"
    venv.EnvBuilder(with_pip=False).create(project / ".venv")
    launcher = project / "启动无黑框本地客户端.vbs"
    shutil.copy2(Path(__file__).resolve().parents[1] / launcher.name, launcher)
    result_file = project / "launched.json"
    (project / "app.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "Path('launched.json').write_text(json.dumps({"
        "'cwd':str(Path.cwd()), 'executable':sys.executable, 'script':sys.argv[0]}))\n",
        encoding="utf-8",
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    subprocess.run(
        ["cscript.exe", "//B", "//NoLogo", str(launcher)],
        cwd=unrelated, check=True, timeout=5, capture_output=True,
    )
    deadline = time.monotonic() + 5
    while not result_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert result_file.is_file(), "launcher did not run app.py from its own folder"
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert Path(result["cwd"]) == project
    assert Path(result["executable"]) == scripts / "pythonw.exe"
    assert Path(result["script"]) == project / "app.py"
