import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path

# Importa path_setup para garantir que a raiz está no sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # ou use import path_setup se quiser centralizar
import path_setup

# Dataset e dataloader
from utils.dataset_utils import ImageStitchingDatasetFiles
from utils.file_utils import descompactar_zip_com_progresso

# Treinamento e checkpoint
from train_loop import train  # ajuste o nome do seu arquivo real aqui

# Importa os modelos básicos que criamos
from discriminator.disc_base import BasicDiscriminator
from generator.gen_base import BasicGenerator

def main():
    teste_name = "_esqueleto_padrao"
    # Descompactar dataset se necessário
    dataset_filename = "dataset_96_64_small_4000.zip"
    epochs = 200

    dataset_dir = "dataset/96x64_small_4000/dataset"
    train_dir = os.path.join(dataset_dir, "train")
    teste_dir = os.path.join("testes", teste_name)
    checkpoint_epoch_dir = os.path.join(teste_dir, "checkpoint_epoch")
    checkpoint_batch_dir = os.path.join(teste_dir, "checkpoint_batch")
    descompactar_zip_com_progresso(os.path.join(dataset_dir, dataset_filename), train_dir)

    # Dataset e DataLoader
    dataset = ImageStitchingDatasetFiles(train_dir, use_gradiente=False)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instanciar modelos básicos
    generator = BasicGenerator().to(device)
    discriminator = BasicDiscriminator().to(device)
    
    # Treinamento
    train(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        epochs=epochs,
        save_every=7200,  # segundos
        checkpoint_dir=checkpoint_epoch_dir,
        checkpoint_batch_dir=checkpoint_batch_dir,
        tensorboard_dir="logs/_esqueleto_padrao",
        lr_g=2e-4,
        lr_d=2e-4,
        lr_min=1e-6,
        gen_steps_mode='fixed',    # pode deixar fixo se quiser ou adaptative
        max_gen_steps=5,              # menor para teste rápido
        vgg_weight=0.5,
        fixeSampleTime=5,             # minutos
        fixed_samples_source=os.path.join(dataset_dir, "fixed_samples.pt"),
        fixed_samples_dest="./fixed_samples",
    )
if __name__ == "__main__":
    main()
