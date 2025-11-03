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

# Funções auxiliares (assumindo que estão nos caminhos corretos)
from utils.carregar_amostras_fixas import carregar_amostras_fixas, salvar_resultados_fixos
from utils.carregar_checkpoint_mais_recente import carregar_checkpoint_mais_recente
from utils.metrics import compute_all_metrics # Assumindo que essa função precisa de fake, target, part1, part2

# Loss Perceptual VGG (copiada para manter o arquivo único)
from torchvision.models import vgg16
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


def train(
    generator, discriminator, dataloader, device, epochs,
    save_every, checkpoint_dir, checkpoint_batch_dir,
    tensorboard_dir, lr_g=2e-4, lr_d=2e-4,
    lr_min=1e-6, gen_steps_mode='adaptive', max_gen_steps=5,
    vgg_weight=0.5, 
    deep_supervision_weight=1.0, 
    fixeSampleTime=5, 
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

    # Schedulers (ajuste dinâmico de LR)
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode="min", factor=0.95, patience=50, min_lr=lr_min
    )
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode="min", factor=0.95, patience=50, min_lr=lr_min
    )

    # Checkpoint
    checkpoint_path, start_epoch, start_batch = carregar_checkpoint_mais_recente(checkpoint_dir, checkpoint_batch_dir)
    if checkpoint_path:
        print(f"\n\nCarregando checkpoint de {checkpoint_path}, epoch {start_epoch}, batch {start_batch}\n\n")
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if start_batch == -1: # Chegou ao final da época anterior, começa a próxima
            start_epoch += 1
            start_batch = 0
    else:
        start_epoch = 0 
        start_batch = 0

    # Inicialização de perdas para o logging
    loss_D = torch.tensor(0.0) # Inicializa para evitar erro no primeiro log
    loss_G = torch.tensor(0.0) # Inicializa para evitar erro no primeiro log
    
    last_checkpoint_time = time.time()
    for epoch in range(start_epoch, epochs):
        total_loss_G_epoch = 0.0 # Renomeado para evitar conflito com loss_G do batch
        total_loss_D_epoch = 0.0 # Renomeado
        total_loss_VGG_epoch = 0.0 # Renomeado
        total_loss_deep_epoch = 0.0 # Renomeado
        total_metrics_epoch = {k: 0.0 for k in ['PSNR', 'SSIM', 'MS-SSIM', 'LPIPS', 'L1', 
                                          'CPU_Usage_%', 'RAM_Usage_MB', 'GPU_Usage_%', 'GPU_Memory_MB']}
        count = 0
        # Histórico de losses para média móvel
        loss_D_hist = []
        loss_G_hist = []
        hist_len = 10  # número de batches para média móvel


        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, ((part1, part2), target) in pbar:
            if epoch == start_epoch and i < start_batch:
                continue

            part1, part2, target = part1.to(device), part2.to(device), target.to(device)

            # ----------------- Treinamento do Discriminator -----------------
            discriminator.train()
            discriminator.zero_grad()
            
            # Concatena as entradas para o discriminador
            real_input = torch.cat([part1, part2, target], dim=1) 
            fake, _ = generator(part1, part2) # <-- AJUSTE: Captura apenas 'fake' e ignora deep_outputs
            fake_input = torch.cat([part1, part2, fake.detach()], dim=1)

            pred_real = discriminator(real_input)
            pred_fake = discriminator(fake_input)

            real_labels = torch.full_like(pred_real, 0.9, device=pred_real.device) # Smooth labeling para reais
            loss_D_real = criterion_GAN(pred_real, real_labels)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0) # Clipping de gradientes
            optimizer_D.step()
            
            # ----------------- Treinamento do Generator -----------------
            # Lógica adaptativa de gen_steps
            if gen_steps_mode == "fixed": # Priorize fixed para testes no mini-dataset
                gen_steps = max_gen_steps
            else:  # adaptativo (revisado para ser mais estável)
                loss_D_hist.append(loss_D.item())
                loss_G_hist.append(loss_G.item()) # Use a loss_G do batch anterior
                if len(loss_D_hist) > hist_len:
                    loss_D_hist.pop(0)
                    loss_G_hist.pop(0)
                
                # Evita divisão por zero ou valores muito pequenos
                mean_loss_D = sum(loss_D_hist) / len(loss_D_hist) if len(loss_D_hist) > 0 else 1.0
                mean_loss_G = sum(loss_G_hist) / len(loss_G_hist) if len(loss_G_hist) > 0 else 1.0

                ratio = mean_loss_G / (mean_loss_D + 1e-8) # Adiciona epsilon para estabilidade

                # Lógica de ajuste de gen_steps mais controlada
                if ratio > 1.5: # Gerador está perdendo muito, precisa de mais passos
                    gen_steps = min(max_gen_steps, gen_steps + 1)
                elif ratio < 0.75: # Discriminador está muito forte, gerador precisa de menos passos para não overfitar
                    gen_steps = max(1, gen_steps - 1)
                else: # Equilíbrio razoável
                    gen_steps = int(max(1, min(max_gen_steps, max_gen_steps / 2))) # Mantém um valor intermediário ou atual
                
                # Garante que gen_steps não vá abaixo de 1
                gen_steps = max(1, gen_steps)

            # Garante que gen_steps não exceda max_gen_steps
            gen_steps = min(gen_steps, max_gen_steps)
            
            for _ in range(gen_steps):
                generator.train()
                generator.zero_grad()
                
                fake, deep_outputs = generator(part1, part2)
                
                # Desabilita o discriminador para o G
                for param in discriminator.parameters():
                    param.requires_grad = False
                
                fake_input = torch.cat([part1, part2, fake], dim=1)
                pred_fake = discriminator(fake_input)
                
                loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_L1 = criterion_L1(fake, target)
                loss_G_VGG = criterion_VGG(fake, target)
                
                if deep_outputs is not None:
                    loss_G_deep = sum(criterion_L1(out, target) for out in deep_outputs)
                else:
                    loss_G_deep = torch.tensor(0.0, device=device)
                
                # --- Ajustes nos pesos das perdas ---
                total_loss = loss_G_GAN + (1.0 * loss_G_L1) + (vgg_weight * loss_G_VGG) + (deep_supervision_weight * loss_G_deep)
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0) # Clipping de gradientes
                optimizer_G.step()
                
                # Re-habilita o discriminador para o próximo passo (mesmo que seja 1 passo do G)
                for param in discriminator.parameters():
                    param.requires_grad = True

            # Atualiza loss_G do batch atual para o próximo cálculo adaptativo
            loss_G = total_loss # Usa a loss_G total calculada neste passo do gerador

            step = epoch * len(dataloader) + i

            # ----------------- Métricas e Logging -----------------
            # Garante que os modelos estão em modo de avaliação para as métricas
            generator.eval()
            discriminator.eval()
            
            # --- Ajustes para compute_all_metrics: passando apenas os tensores necessários ---
            metrics_dict = compute_all_metrics(fake.detach(), target.detach(), part1.detach(), part2.detach(), writer, step)

            # Adiciona LRs e gen_steps
            metrics_dict['LR_G'] = optimizer_G.param_groups[0]['lr']
            metrics_dict['LR_D'] = optimizer_D.param_groups[0]['lr']
            metrics_dict['LR_G_max'] = lr_g
            metrics_dict['LR_G_min'] = lr_min
            metrics_dict['LR_D_max'] = lr_d
            metrics_dict['LR_D_min'] = lr_min
            metrics_dict['Gen_Steps'] = gen_steps

            # Adiciona losses
            metrics_dict['Loss_GAN'] = loss_G_GAN.item()
            metrics_dict['Loss_L1'] = loss_G_L1.item()
            metrics_dict['Loss_VGG'] = loss_G_VGG.item()
            metrics_dict['Loss_Deep'] = loss_G_deep.item()
            metrics_dict['Loss_Generator'] = total_loss.item()
            metrics_dict['Loss_Discriminator'] = loss_D.item()

            # Grava tudo no TensorBoard
            for key, value in metrics_dict.items():
                if value is not None:
                    writer.add_scalar(f"Step/{key}", value, step)

            total_loss_G_epoch += total_loss.item()
            total_loss_D_epoch += loss_D.item()
            total_loss_VGG_epoch += loss_G_VGG.item()
            total_loss_deep_epoch += loss_G_deep.item()
            count += 1
            
            # Atualiza métricas para logging da época
            for k in total_metrics_epoch:
                if k in metrics_dict and metrics_dict[k] is not None:
                    total_metrics_epoch[k] += metrics_dict[k]


            pbar.set_postfix({"loss_G": f"{total_loss.item():.4f}", "loss_D": f"{loss_D.item():.4f}"})

            # ----------------- Salvamento de amostras fixas -----------------
            if datetime.now() - last_fixed_sample_time >= timedelta(minutes=fixeSampleTime):
                # Certifica-se que o gerador está em modo de avaliação para salvar amostras
                generator.eval()
                salvar_resultados_fixos(generator, fixed_samples, output_dir=fixed_samples_dest, step_tag=f"{epoch}_{i}")
                last_fixed_sample_time = datetime.now()
                generator.train() # Retorna ao modo de treino

            # ----------------- Checkpoint por tempo -----------------
            if time.time() - last_checkpoint_time > save_every:
                listaArquivos = os.listdir(checkpoint_batch_dir)
                if len(listaArquivos) >= 5:     # mantém no máximo 5 arquivos
                    listaArquivos = sorted(listaArquivos, key=lambda x: os.path.getmtime(os.path.join(checkpoint_batch_dir, x)))
                    arquivo_remover = listaArquivos[0]
                    os.remove(os.path.join(checkpoint_batch_dir, arquivo_remover))
                    print(f"Removido checkpoint antigo: {arquivo_remover}")
                print(f"\nSalvando checkpoint de batch em epoch {epoch}, batch {i} em {checkpoint_batch_dir}")
                torch.save({
                    'epoch': epoch,
                    'batch': i,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(), # Salvar otimizadores
                    'optimizer_D_state_dict': optimizer_D.state_dict(), # Salvar otimizadores
                    'scheduler_G_state_dict': scheduler_G.state_dict(), # Salvar schedulers
                    'scheduler_D_state_dict': scheduler_D.state_dict(), # Salvar schedulers
                }, os.path.join(checkpoint_batch_dir, f'checkpoint_epoch{epoch}_batch{i}.pt'))
                last_checkpoint_time = time.time()
            
            # --- no final de cada batch (após losses) ---
            # Schedulers são chamados com base nas perdas do gerador e discriminador
            scheduler_G.step(total_loss.item())
            scheduler_D.step(loss_D.item())


        # ----------------- Logging por época -----------------
        if count > 0: # Evitar divisão por zero se a época não tiver batches
            writer.add_scalar("Epoch/Loss_Generator", total_loss_G_epoch / count, epoch)
            writer.add_scalar("Epoch/Loss_Discriminator", total_loss_D_epoch / count, epoch)
            writer.add_scalar("Epoch/Loss_VGG", total_loss_VGG_epoch / count, epoch)
            writer.add_scalar("Epoch/Loss_Deep", total_loss_deep_epoch / count, epoch)
            for k, v in total_metrics_epoch.items():
                writer.add_scalar(f"Epoch/Metrics/{k}", v / count, epoch)

        # Salva checkpoint por época
        print(f"\nSalvando checkpoint de época {epoch} em {checkpoint_dir}")
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt'))

    writer.close()