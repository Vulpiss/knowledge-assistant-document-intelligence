# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import runpy

from PyInstaller.utils.hooks import (
    collect_all,
    collect_submodules,
    copy_metadata,
)


project_root = Path(SPECPATH).resolve().parent
model_helper = runpy.run_path(
    str(project_root / "packaging" / "prepare_embedding_model.py")
)
embedding_model_path = model_helper["prepare_embedding_model"](
    local_files_only=True
)
embedding_model_files = model_helper["MODEL_FILES"]

datas = [
    (
        str(project_root / "app" / "interfaces" / "streamlit_app.py"),
        "app/interfaces",
    ),
    (
        str(project_root / ".streamlit" / "config.toml"),
        ".streamlit",
    ),
    (str(project_root / "README.md"), "."),
    (str(project_root / "VERSION"), "."),
]

for relative_model_file in embedding_model_files:
    relative_path = Path(relative_model_file)
    destination = (
        Path("models")
        / "paraphrase-multilingual-MiniLM-L12-v2"
        / relative_path.parent
    )
    datas.append(
        (
            str(embedding_model_path / relative_path),
            str(destination),
        )
    )

binaries = []
hiddenimports = collect_submodules("app")

for package_name in (
    "streamlit",
    "sentence_transformers",
    "qdrant_client",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

hiddenimports += collect_submodules("transformers.models.bert")

for distribution_name in (
    "streamlit",
    "sentence-transformers",
    "transformers",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
    "torch",
    "scikit-learn",
    "scipy",
    "numpy",
    "qdrant-client",
    "pydantic",
    "python-dotenv",
):
    try:
        datas += copy_metadata(distribution_name)
    except Exception:
        pass

a = Analysis(
    [str(project_root / "desktop_launcher.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "tensorboard",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="KnowledgeAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="KnowledgeAssistant",
)
