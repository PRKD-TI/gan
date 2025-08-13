# ------------------ dataset utilities ------------------
# add all imports here
import os
import torch
import torchvision.utils as vutils


def carregar_amostras_fixas(train_loader, caminho='fixed_samples.pt', device='cuda'):
    if os.path.exists(caminho):
        print(f'[INFO] Carregando amostras fixas de {caminho}')
        fixed_samples = torch.load(caminho)
    else:
        print('[INFO] Selecionando novas amostras fixas...')
        fixed_samples = []
        for i, ((p1, p2), gt) in enumerate(train_loader):
            if i >= 5:
                break
            fixed_samples.append(((p1[0].unsqueeze(0), p2[0].unsqueeze(0)), gt[0].unsqueeze(0)))
        torch.save(fixed_samples, caminho)
        print(f'[INFO] Amostras fixas salvas em {caminho}')
    fixed_samples = [((p1.to(device), p2.to(device)), gt.to(device)) for ((p1, p2), gt) in fixed_samples]
    return fixed_samples

# usa EMA se for tupla (generator, ema)
def salvar_resultados_fixos(generator, fixed_samples, output_dir, step_tag):
    if isinstance(generator, tuple):
        gen_for_eval = generator[1]
    else:
        gen_for_eval = generator
    gen_for_eval.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, ((p1, p2), gt) in enumerate(fixed_samples):
            fake = gen_for_eval(p1, p2)
            linha = torch.cat([p1, p2, fake, gt], dim=3)
            vutils.save_image(linha, os.path.join(output_dir, f"sample_{idx+1}_{step_tag}.png"), normalize=True)
    if not isinstance(generator, tuple):
        gen_for_eval.train()

