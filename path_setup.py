import sys
from pathlib import Path

# Adiciona a pasta raiz do projeto no sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# from pathlib import Path

# def print_tree(root: Path, prefix: str = ""):
#     """Imprime árvore de diretórios."""
#     files = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
#     for idx, path in enumerate(files):
#         connector = "└── " if idx == len(files) - 1 else "├── "
#         print(prefix + connector + path.name)
#         if path.is_dir():
#             extension = "    " if idx == len(files) - 1 else "│   "
#             print_tree(path, prefix + extension)

# if __name__ == "__main__":
#     raiz = Path(__file__).resolve().parent  # ajusta se quiser outro ponto inicial
#     print_tree(raiz)
