import sys
from pathlib import Path
import os
import time
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm

from utils.carregar_amostras_fixas import carregar_amostras_fixas, salvar_resultados_fixos
from utils.carregar_checkpoint_mais_recente import carregar_checkpoint_mais_recente
from utils.metrics import compute_all_metrics, compute_gradient_penalty

from torchvision.models import vgg16
import lpips

# --- Loss Perceptual VGG ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights='DEFAULT').features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)
        return F.l1_loss(input_vgg, target_vgg)

# LPIPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(device)

# ------------------ Treinamento ------------------
def train(
    generator, discriminator, dataloader, device, epochs,
    save_every, checkpoint_dir, checkpoint_batch_dir,
    tensorboard_dir, lr_g=2e-4, lr_d=2e-4,
    lr_min=1e-6, gen_steps_mode='adaptive', max_gen_steps=5,
    vgg_weight=0.5, fixeSampleTime=5,  # minutos
    fixed_samples_source='../fixed_samples.pt',
    fixed_samples_dest='./fixed_samples'
):
    # Carrega amostras fixas para visualização
    fixed_samples = carregar_amostras_fixas(dataloader, caminho=fixed_samples_source, device=device)
    last_fixed_sample_time = datetime.now()

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_batch_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_VGG = VGGPerceptualLoss().to(device)

    # Otimizadores
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # Checkpoint
    checkpoint_path, start_epoch, start_batch = carregar_checkpoint_mais_recente(checkpoint_dir, checkpoint_batch_dir)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if start_batch == -1:
            start_epoch += 1
            start_batch = 0
    else:
        start_epoch = 0 
        start_batch = 0

    last_checkpoint_time = time.time()

    for epoch in range(start_epoch, epochs):
        total_loss_G = 0.0
        total_loss_D = 0.0
        total_loss_VGG = 0.0
        total_metrics = {k: 0.0 for k in ['PSNR', 'SSIM', 'MS-SSIM', 'LPIPS', 'L1', 
                                          'CPU_Usage_%', 'RAM_Usage_MB', 'GPU_Usage_%', 'GPU_Memory_MB']}
        count = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, ((part1, part2), target) in pbar:
            if epoch == start_epoch and i < start_batch:
                continue

            part1, part2, target = part1.to(device), part2.to(device), target.to(device)

            # ----------------- Treinamento do Discriminator -----------------
            optimizer_D.zero_grad()
            real_input = torch.cat([part1, part2, target], dim=1)
            fake = generator(part1, part2)
            fake_input = torch.cat([part1, part2, fake.detach()], dim=1)

            pred_real = discriminator(real_input)
            pred_fake = discriminator(fake_input)

            real_labels = torch.full_like(pred_real, 0.9, device=pred_real.device)
            loss_D_real = criterion_GAN(pred_real, real_labels)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()

            # ----------------- Treinamento do Generator -----------------
            gen_steps = max_gen_steps if gen_steps_mode=='fixed' else max_gen_steps  # placeholder para adaptativo
            for _ in range(gen_steps):
                optimizer_G.zero_grad()
                fake = generator(part1, part2)
                fake_input = torch.cat([part1, part2, fake], dim=1)
                pred_fake = discriminator(fake_input)

                loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_L1 = criterion_L1(fake, target)
                loss_G_VGG = criterion_VGG(fake, target)
                loss_G = 8.0 * loss_G_GAN + 2.0 * loss_G_L1 + vgg_weight * loss_G_VGG
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()

            step = epoch * len(dataloader) + i

            # ----------------- Métricas e Logging -----------------
            metrics_dict = compute_all_metrics(fake, target, part1, part2, writer, step)

            # Adiciona LR atuais e limites
            metrics_dict['LR_G'] = optimizer_G.param_groups[0]['lr']
            metrics_dict['LR_D'] = optimizer_D.param_groups[0]['lr']
            metrics_dict['LR_G_max'] = lr_g
            metrics_dict['LR_G_min'] = lr_min
            metrics_dict['LR_D_max'] = lr_d
            metrics_dict['LR_D_min'] = lr_min

            # Adiciona losses
            metrics_dict['Loss_GAN'] = loss_G_GAN.item()
            metrics_dict['Loss_L1'] = loss_G_L1.item()
            metrics_dict['Loss_VGG'] = loss_G_VGG.item()
            metrics_dict['Loss_Generator'] = loss_G.item()
            metrics_dict['Loss_Discriminator'] = loss_D.item()

            # Grava tudo no TensorBoard
            for key, value in metrics_dict.items():
                if value is not None:
                    writer.add_scalar(f"Step/{key}", value, step)

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            total_loss_VGG += loss_G_VGG.item()
            count += 1

            pbar.set_postfix({"loss_G": f"{loss_G.item():.4f}", "loss_D": f"{loss_D.item():.4f}"})

            # ----------------- Salvamento de amostras fixas -----------------
            if datetime.now() - last_fixed_sample_time >= timedelta(minutes=fixeSampleTime):
                salvar_resultados_fixos(generator, fixed_samples, output_dir=fixed_samples_dest, step_tag=f"{epoch}_{step}")
                last_fixed_sample_time = datetime.now()

            # ----------------- Checkpoint por tempo -----------------
            if time.time() - last_checkpoint_time > save_every:
                torch.save({
                    'epoch': epoch,
                    'batch': i,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                }, os.path.join(checkpoint_batch_dir, f'checkpoint_epoch{epoch}_batch{i}.pt'))
                last_checkpoint_time = time.time()

        # ----------------- Logging por época -----------------
        writer.add_scalar("Epoch/Loss_Generator", total_loss_G / count, epoch)
        writer.add_scalar("Epoch/Loss_Discriminator", total_loss_D / count, epoch)
        writer.add_scalar("Epoch/Loss_VGG", total_loss_VGG / count, epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f"Epoch/Metrics/{k}", v / count, epoch)

        # Salva checkpoint por época
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt'))

    writer.close()
