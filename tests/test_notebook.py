"""The Colab notebook must be valid JSON, compile, and embed EXACTLY the repository's training code."""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "colab" / "Plant_Disease_Classification_Training.ipynb"
EMBEDDED = ["src/constants.py", "src/preprocessing.py", "src/model_builder.py", "src/dataset_utils.py",
            "src/dataset_download.py", "training/prepare_dataset.py", "training/evaluate.py", "training/train.py"]


def load():
    return json.loads(NB.read_text(encoding="utf-8"))


def src(cell):
    return "".join(cell["source"])


def test_notebook_is_valid_nbformat4():
    nb = load()
    assert nb["nbformat"] == 4 and isinstance(nb["cells"], list) and len(nb["cells"]) > 10
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown") and isinstance(c["source"], list)
        if c["cell_type"] == "code":
            assert "outputs" in c and "execution_count" in c


def test_embedded_files_match_repository_sources():
    cells = {}
    for c in load()["cells"]:
        m = re.match(r"%%writefile /content/pdc/(\S+)\n", src(c))
        if m:
            cells[m.group(1)] = src(c).split("\n", 1)[1]
    assert sorted(cells) == sorted(EMBEDDED)
    for rel in EMBEDDED:
        assert cells[rel].rstrip("\n") == (ROOT / rel).read_text(encoding="utf-8").rstrip("\n"), (
            f"{rel} differs from the notebook; run: python scripts/build_notebook.py")


def test_all_code_cells_are_valid_python():
    for i, c in enumerate(load()["cells"]):
        if c["cell_type"] != "code":
            continue
        text = src(c)
        if text.startswith("%%writefile"):
            text = text.split("\n", 1)[1]
        ast.parse(text, filename=f"cell {i}")


def test_notebook_uses_same_training_entrypoints_and_no_fake_results():
    text = "\n".join(src(c) for c in load()["cells"])
    assert "training/train.py" in text and "training/prepare_dataset.py" in text
    assert "evaluation_results.json" in text
    assert "random.random" not in text and "np.random.rand" not in text


def test_embedded_scripts_run_from_a_clean_colab_like_layout(tmp_path):
    """Recreate the Colab file layout from the notebook alone and ask train.py for --help."""
    for c in load()["cells"]:
        m = re.match(r"%%writefile /content/pdc/(\S+)\n", src(c))
        if m:
            f = tmp_path / m.group(1)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(src(c).split("\n", 1)[1], encoding="utf-8")
    (tmp_path / "src" / "__init__.py").touch()
    (tmp_path / "training" / "__init__.py").touch()
    r = subprocess.run([sys.executable, str(tmp_path / "training" / "train.py"), "--help"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "--fine-tune-epochs" in r.stdout, r.stderr[-500:]
