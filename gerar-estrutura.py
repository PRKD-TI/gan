from pathlib import Path

# Diretório base (pode ser alterado)
base_dir = Path(".")

# Estrutura de diretórios e arquivos
structure = {
    "generator": [
        "gen_base.py",
        "gen_unet.py",
        "gen_resnet.py"
    ],
    "discriminator": [
        "disc_base.py",
        "disc_patchgan.py",
        "disc_multiscale.py"
    ],
    "attention": [
        "cbam.py",
        "self_attention.py",
        "spatial_attention.py"
    ],
    "dataset": [
        # Datasets terão subpastas específicas, por exemplo: 96x64_dataset
    ],
    "util": [
        "metrics.py",
        "dataset_utils.py",
        "tensorboard_utils.py",
        "file_utils.py"
    ],
    "testes": [
        # Cada teste terá subpastas próprias
    ],
    "logs": []  # Pasta para logs do TensorBoard
}

# Estrutura inicial para testes
test_names = ["gan_simples_1", "gan_simples_2", "gan_simples_3"]
test_subfolders = ["checkpoint_epoch", "checkpoint_batch", "fixed_samples"]

def create_structure():
    for folder, files in structure.items():
        folder_path = base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            file_path = folder_path / file
            if not file_path.exists():
                file_path.write_text("# " + file.replace(".py", "").replace("_", " ").title() + "\n")

    # Criar estrutura para testes
    testes_path = base_dir / "testes"
    for test in test_names:
        test_path = testes_path / test
        test_path.mkdir(parents=True, exist_ok=True)
        
        # Criar subpastas
        for sub in test_subfolders:
            (test_path / sub).mkdir(parents=True, exist_ok=True)
        
        # Criar arquivos básicos de treinamento
        train_file = test_path / "train_loop.py"
        if not train_file.exists():
            train_file.write_text("# Loop de treinamento para " + test + "\n")

        main_file = test_path / "main.py"
        if not main_file.exists():
            main_file.write_text("# Arquivo principal para executar o treinamento de " + test + "\n")

    print("Estrutura de diretórios e arquivos criada com sucesso!")

if __name__ == "__main__":
    create_structure()
