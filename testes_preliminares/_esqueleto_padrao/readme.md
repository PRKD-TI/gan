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

O gerador inicial (`BasicGenerator`) é baseado em uma **arquitetura Encoder-Decoder simples** que recebe duas imagens (`part1`, `part2`) e concatena internamente.

- Entrada: `part1` (RGB), `part2` (RGB)
- Concatenação: `[B, 3, H, W] + [B, 3, H, W] -> [B, 6, H, W]`
- Encoder: 3 camadas de convolução com downsampling
- Bottleneck: camada profunda para aprendizado de contexto
- Decoder: 3 camadas de `ConvTranspose` para upsampling
- Saída final: imagem RGB `[B, 3, H, W]` com ativação `Tanh`

Esse modelo serve apenas como **baseline experimental**.

### Exemplo de uso

```python
import torch
from generator.gen_base import BasicGenerator

# Criar instância do gerador
generator = BasicGenerator()

# Gerar imagens falsas a partir de partes artificiais
part1 = torch.randn(1, 3, 64, 96)
part2 = torch.randn(1, 3, 64, 96)
fake = generator(part1, part2)

print(fake.shape)  # -> torch.Size([1, 3, 64, 96])
```

---

## 🕵️ Discriminador - `discriminator/disc_base.py`

O discriminador inicial (`BasicDiscriminator`) segue a ideia de um **PatchGAN condicional**.

- Entrada:

  - `part1` (RGB)
  - `part2` (RGB)
  - imagem real ou gerada (RGB)
  - Concatenados: `[B, 9, H, W]`

- Arquitetura:

  - 4 camadas convolucionais progressivamente mais profundas (64 → 512 filtros)
  - BatchNorm a partir da 2ª camada
  - Ativação **LeakyReLU(0.2)** em todas as camadas
  - Camada final `Conv2d(512 → 1)` gera o mapa de real/falso

- Saída:

  - Um **mapa de patches** `[B, 1, H/16, W/16]`
  - Cada valor representa a confiança de real/falso em um patch da imagem

Esse modelo fornece supervisão local, forçando o gerador a criar detalhes realistas em regiões específicas.

### Exemplo de uso

```python
import torch
from discriminator.disc_base import BasicDiscriminator

# Criar instância do discriminador
discriminator = BasicDiscriminator()

# Partes e imagem fake (mesmo tamanho)
part1 = torch.randn(1, 3, 64, 96)
part2 = torch.randn(1, 3, 64, 96)
fake = torch.randn(1, 3, 64, 96)

# Concatenar para o discriminador
inp = torch.cat([part1, part2, fake], dim=1)
out = discriminator(inp)

print(out.shape)  # -> torch.Size([1, 1, 7, 11]) (exemplo)
```

---

## 🔁 Loop de Treinamento - `train_loop.py`

O loop de treinamento implementa a lógica básica:

1. **Forward Generator**

   ```python
   fake = generator(part1, part2)
   ```

2. **Treino do Discriminador**

   - Real loss (com ground truth)
   - Fake loss (com imagens geradas)

3. **Treino do Gerador**

   - Objetivo: enganar o discriminador (adversarial loss)
   - Inclui **L1 Loss** para proximidade com a ground truth

4. **Zerar gradientes corretamente**:

   ```python
   optimizer_G.zero_grad()
   optimizer_D.zero_grad()
   ```

5. **Otimizadores**

   - Ambos usam **Adam** (`lr=0.0002, betas=(0.5, 0.999)`)

---

## 📊 Métricas

Para este primeiro teste, métricas foram simplificadas:

- `Loss_G`: perda do gerador (adversarial + L1)
- `Loss_D`: perda do discriminador (real + fake)

Futuramente podemos incluir:

- PSNR, SSIM, MS-SSIM
- LPIPS

---

## 🚀 Próximos Passos

1. Validar o treinamento com dataset reduzido
2. Visualizar algumas amostras geradas
3. Incluir métricas de qualidade
4. Evoluir para arquiteturas mais profundas (UNet, DualEncoder, atenção, etc.)

---

