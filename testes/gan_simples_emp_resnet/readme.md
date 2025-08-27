# 🧪 Primeiro Teste - Esqueleto Padrão da GAN

Este repositório contém a estrutura mínima de uma **GAN para costura de imagens**, pensada como esqueleto de testes iniciais. A ideia é ter uma base simples, limpa e fácil de expandir.

---

## 📂 Estrutura de Diretórios

```
.
├── generator/
│   └── gen_base.py          # Gerador base (Encoder-Decoder simples)
├── discriminator/
│   └── disc_base.py         # Discriminador base (PatchGAN simples)
├── testes/
│   └── _esqueleto_padrao/
│       ├── train.py         # Script principal de treino
│       └── train_loop.py    # Loop de treinamento
└── utils/
    └── dataset_utils.py     # Dataset customizado para costura de imagens
```

---

## 🎨 Gerador - `generator/gen_base.py`

O gerador inicial (`BasicGenerator`) é baseado em uma **arquitetura Encoder-Decoder** que recebe duas imagens (`part1`, `part2`) e concatena internamente.

**Arquitetura:**

- **Entrada:** `part1` (RGB), `part2` (RGB) → concatenação `[B, 6, H, W]`
- **Encoder:** 3 camadas convolucionais com downsampling progressivo (6 → 64 → 128 → 256)
- **Bottleneck:** camada convolucional profunda (256 → 512) para capturar contexto global
- **Decoder:** 3 camadas `ConvTranspose2d` para upsampling (512 → 256 → 128 → 64)
- **Saída:** imagem RGB `[B, 3, H, W]` com ativação `Tanh` entre -1 e 1

**Observação:** Este gerador serve como baseline experimental, fácil de substituir por arquiteturas mais complexas no futuro.

---

## 🕵️ Discriminador - `discriminator/disc_base.py`

O discriminador inicial (`BasicDiscriminator`) segue a ideia de um **PatchGAN condicional**.

**Arquitetura:**

- **Entrada:** concatenação de `part1`, `part2` e imagem real/fake → `[B, 9, H, W]`
- 4 camadas convolucionais progressivamente mais profundas (64 → 512 filtros)
- BatchNorm aplicado a partir da 2ª camada
- Ativação **LeakyReLU(0.2)** em todas as camadas
- Camada final `Conv2d(512 → 1)` produz mapa de confiança para cada patch da imagem

**Saída:** mapa de patches `[B, 1, H/16, W/16]` representando real/falso em cada região da imagem.

---

## 🔁 Loop de Treinamento - `train_loop.py`

O loop implementa a lógica completa do treino da GAN:

1. **Learning Rates Fixos**

   - Adam com `lr_g=0.0002` e `lr_d=0.0002`
   - Mantidos constantes durante todas as épocas

2. **Forward do Gerador**

   - Recebe `part1` e `part2`, concatena internamente
   - Gera imagem fake para o discriminador

3. **Treino do Discriminador**

   - Entrada real: concatenação `[part1, part2, target]`
   - Entrada fake: concatenação `[part1, part2, fake.detach()]`
   - Loss: média de `loss_real` e `loss_fake` usando `BCEWithLogitsLoss`
   - Gradient clipping aplicado para estabilidade

4. **Treino do Gerador**

   - Objetivo: enganar o discriminador (`loss_GAN`) + proximidade com ground truth (`L1 Loss`) + similaridade perceptual (`VGG Loss`)
   - **VGG Perceptual Loss:** compara ativação de camadas intermediárias do VGG16
   - **LPIPS:** métrica perceptual adicional usada para monitoramento
   - Loss total: `loss_G = 8.0 * loss_GAN + 2.0 * L1 + vgg_weight * loss_VGG`

5. **Amostras Fixas**

   - Salva imagens geradas de um subset fixo a cada `fixeSampleTime` minutos

6. **Checkpoints**

   - Salvamento por batch e por época, permitindo reinício do treinamento

7. **TensorBoard**

   - Monitoramento de losses (`GAN`, `L1`, `VGG`, `Generator`, `Discriminator`) e métricas perceptuais (`LPIPS`, PSNR, SSIM, MS-SSIM, L1)

**Observações:**

- GenSteps atualmente fixos (`max_gen_steps`) para simplificar o primeiro teste.
- Gradientes do gerador e discriminador são clipados para `max_norm=1.0`.
- O loop está preparado para futuramente implementar **gen\_steps adaptativo** ou **learning rate scheduler**.

---

## 📊 Métricas Monitoradas

- `Loss_G`: perda total do gerador (GAN + L1 + VGG)
- `Loss_D`: perda do discriminador (real + fake)
- `Loss_VGG`: componente perceptual do VGG
- `LPIPS`: métrica perceptual
- `PSNR`, `SSIM`, `MS-SSIM`, `L1`
- Monitoramento de uso de CPU, RAM e GPU

---

## 🚀 Próximos Passos

1. Validar o treinamento com dataset reduzido
2. Visualizar algumas amostras geradas
3. Explorar learning rates dinâmicos e gen\_steps adaptativos
4. Evoluir para arquiteturas mais profundas (UNet, DualEncoder, atenção, etc.)
5. Incluir métricas adicionais para avaliação quantitativa

