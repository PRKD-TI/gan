import os
import warnings
from pathlib import Path

# Silencia warnings chatos do Torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
from torch.utils.data import DataLoader

# Garante que a raiz do projeto está no sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(ROOT_DIR))

import path_setup  # inicialização de paths (seu arquivo auxiliar)

# Imports locais
from utils.dataset_utils import ImageStitchingDatasetFiles
from utils.file_utils import descompactar_zip_com_progresso
from train_loop import train
from generator_preliminares.gen_dual_unet_deep6_skip_att_bottleneck_att_encdec_att_skip import DualEncoderUNetSkip6_Attn as default_generator
from discriminator.disc_patchgan import PatchDiscriminator as default_discriminator

def main():
    # === Configurações do experimento ===
    teste_name = Path(__file__).resolve().parent.name

    # Descompactar dataset se necessário
    dataset_name = "256x128_small_4000"                   # Vai definir nome das pastas também
    dataset_filename = dataset_name + ".zip"
    dataset_dir = ROOT_DIR / "dataset" / dataset_name / "dataset"
    train_dir = dataset_dir / "train"
    epochs = 10

    # Pastas de saída
    teste_dir = ROOT_DIR / "testes" / teste_name
    checkpoint_epoch_dir = teste_dir / f"checkpoint_epoch/{dataset_name}"
    checkpoint_batch_dir = teste_dir / f"checkpoint_batch/{dataset_name}"

    print("Checkpoint epoch dir:", checkpoint_epoch_dir)
    print("Checkpoint batch dir:", checkpoint_batch_dir)


    # os.makedirs(checkpoint_epoch_dir, exist_ok=True)
    # os.makedirs(checkpoint_batch_dir, exist_ok=True)

    # Descompactar dataset, se necessário
    descompactar_zip_com_progresso(dataset_dir / dataset_filename, train_dir)

    # === Dataset e DataLoader ===
    dataset = ImageStitchingDatasetFiles(str(train_dir), use_gradiente=False)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )

    # === Dispositivo ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Modelos ===
    generator = default_generator(input_nc=3, output_nc=3).to(device)  # <<< UNet dual encoder
    discriminator = default_discriminator(input_nc=9).to(device)          # PatchGAN igual antes

    # === Treinamento ===
    train(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        epochs=epochs,
        save_every=7200,  # segundos
        checkpoint_dir=str(checkpoint_epoch_dir),
        checkpoint_batch_dir=str(checkpoint_batch_dir),
        tensorboard_dir=str(ROOT_DIR / "logs" / "prkd" / (teste_name + f"_{dataset_name}")),
        lr_g=2e-4,
        lr_d=2e-4,
        lr_min=1e-6,
        gen_steps_mode="fixed",   # ou 'adaptive'
        max_gen_steps=5,          # menor p/ testes rápidos
        vgg_weight=0.5,
        fixeSampleTime=5,         # minutos
        fixed_samples_source=str(dataset_dir / "fixed_samples.pt"),
        fixed_samples_dest=str(teste_dir / "fixed_samples" / dataset_name),
    )


if __name__ == "__main__":
    main()
