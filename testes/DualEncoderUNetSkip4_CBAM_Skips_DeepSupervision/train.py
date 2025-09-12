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
# Importe o gerador atualizado (DualEncoderUNet_Res_ASPP_CBAM_DeepSupervision)
from generator.DualEncoderUNetSkip4_CBAM_Skips_DeepSupervision import DualEncoderUNet_Res_ASPP_CBAM_DeepSupervision as default_generator
from discriminator.disc_patchgan import PatchDiscriminator as default_discriminator

def main():
    # === Configurações do experimento ===
    teste_name = Path(__file__).resolve().parent.name

    # Descompactar dataset se necessário
    dataset_name = "mini_dataset_512x256" # <--- NOVO NOME para o mini-dataset
    # Assumindo que você criou o mini-dataset dentro da pasta 'dataset/mini_dataset_512x256'
    dataset_dir = ROOT_DIR / "dataset" / dataset_name
    train_dir = dataset_dir / "train"
    epochs = 100 # <--- AUMENTADO para testes em dataset pequeno

    # Pastas de saída - Ajustadas para o mini-dataset
    teste_dir = ROOT_DIR / "testes" / teste_name # Mantém a estrutura de "teste_name"
    checkpoint_epoch_dir = teste_dir / "checkpoint_epoch" / dataset_name
    checkpoint_batch_dir = teste_dir / "checkpoint_batch" / dataset_name
    fixed_samples_dest = teste_dir / "fixed_samples" / dataset_name
    tensorboard_dir = ROOT_DIR / "logs" / "prkd" / (teste_name + f"_{dataset_name}")

    print("Checkpoint epoch dir:", checkpoint_epoch_dir)
    print("Checkpoint batch dir:", checkpoint_batch_dir)


    # Descompactar dataset, se necessário (Para o mini-dataset, pode ser manual ou um script)
    # Aqui, a lógica de descompactar zip é mantida, mas certifique-se que o "mini_dataset_512x256.zip" exista
    dataset_filename = dataset_name + ".zip" 
    descompactar_zip_com_progresso(dataset_dir / dataset_filename, train_dir)


    # === Dataset e DataLoader ===
    dataset = ImageStitchingDatasetFiles(str(train_dir), use_gradiente=False)
    dataloader = DataLoader(
        dataset,
        batch_size=4,       # <--- REDUZIDO para P2000 e testes
        shuffle=True,
        num_workers=4,      # <--- REDUZIDO para P2000 e testes
        prefetch_factor=2,  # <--- REDUZIDO para P2000 e testes
        pin_memory=True,
    )

    # === Dispositivo ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Modelos ===
    generator = default_generator(
        input_nc=3, 
        output_nc=3, 
        ngf=32, 
        use_cbam=True, 
        use_res=True, 
        use_non_local=False # <--- Desativado por padrão para iniciar. Ative se a P2000 aguentar e precisar.
    ).to(device) 
    discriminator = default_discriminator(input_nc=9).to(device)

    # === Treinamento ===
    train(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        epochs=epochs,
        save_every=1800,  # segundos (30 minutos) - para evitar muitos checkpoints em testes rápidos
        checkpoint_dir=str(checkpoint_epoch_dir),
        checkpoint_batch_dir=str(checkpoint_batch_dir),
        tensorboard_dir=str(tensorboard_dir), # Já configurado acima
        lr_g=2e-4,
        lr_d=2e-4,
        lr_min=1e-6, # Mantido, mas pode ir para 1e-7 se LRs caírem muito
        gen_steps_mode="fixed",   # <--- ALTERADO para 'fixed' para testes iniciais
        max_gen_steps=1,          # <--- ALTERADO para 1 para começar. Pode ir para 2 ou 3 se estabilizar.
        vgg_weight=0.5,           # <--- Sugestão: Aumente para 1.0 ou 2.0 se imagens estiverem borradas
        deep_supervision_weight=0.3, # <--- Sugestão: Aumente para 0.5 ou 1.0 se as saídas intermediárias forem ruins
        fixeSampleTime=1,         # minutos - Para ver a evolução mais de perto
        fixed_samples_source=str(dataset_dir / "fixed_samples.pt"), # Caminho para amostras fixas do mini-dataset
        fixed_samples_dest=str(fixed_samples_dest), # Caminho para salvar amostras fixas
    )


if __name__ == "__main__":
    main()