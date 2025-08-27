
# 🧩 UNet Empilhada para Costura de Imagens

Este projeto implementa uma **UNet empilhada** como gerador para uma **GAN de costura de imagens**.  
A ideia central é **empilhar as duas partes da imagem** (parte1 e parte2) como canais de entrada, resultando em uma entrada de **6 canais** (2 × 3 RGB).

---

## 📐 Arquitetura

- **Entrada:**  
  - Parte1: (3×H×W)  
  - Parte2: (3×H×W)  
  - Concatenação → (6×H×W)

- **Encoder:**  
  - Três blocos Conv+BN+ReLU que reduzem progressivamente a resolução
  - Extraem características profundas da imagem

- **Bottleneck:**  
  - Camada convolucional no fundo da rede
  - Condensa informações globais

- **Decoder:**  
  - Três blocos UpConv + Concat com *skip connections*
  - Reconstrói a imagem de saída a partir das features comprimidas

- **Saída:**  
  - Imagem reconstruída (3×H×W)  
  - Ativação final: **Tanh**

---

## 🔗 Conexões de Skip

A UNet utiliza conexões *skip* entre os encoders e os decoders correspondentes, permitindo:

- Preservar detalhes de baixa frequência
- Facilitar o aprendizado da reconstrução espacial

---

## 🖼️ Diagrama da Arquitetura

![Diagrama UNet Empilhada](unet_stack_diagram.png)

---

## ⚖️ Uso na GAN

- **Gerador (G):** UNet empilhada (entrada: 6 canais, saída: 3 canais)  
- **Discriminador (D):** PatchGAN (entrada: imagem gerada ou groundtruth empilhada → 9 canais)  
- **Loss:**  
  - Adversarial (GAN Loss)  
  - L1 Loss  
  - LPIPS (perceptual loss)

---

## 📂 Estrutura do Projeto

```
gan/
 ├── generator/
 │    └── gen_unet_stack.py   # Implementação da UNet empilhada
 ├── discriminator/
 │    └── disc_patchgan.py    # Discriminador PatchGAN
 ├── testes/
 │    └── gan_simples_emp_unet/  # Experimentos com UNet
 └── dataset/                 # Dados de treino
```

---

## 🚀 Execução

```bash
cd testes/gan_simples_emp_unet
python train.py
```

Isso iniciará o treinamento da GAN com **UNet empilhada** como gerador.
