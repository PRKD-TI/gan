"""
Deep Supervision para Decoder
-----------------------------
Funções para aplicar supervisionamento intermediário em múltiplos níveis do decoder.
Isso ajuda a melhorar o fluxo de gradiente, acelera convergência e reforça aprendizado em resoluções intermediárias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_deep_supervision_outputs(decoder_outputs, final_channels=3):
    """
    Gera saídas de deep supervision a partir de múltiplos níveis do decoder.
    
    Args:
        decoder_outputs (list of torch.Tensor): Lista de tensores do decoder em diferentes resoluções.
        final_channels (int): Número de canais da imagem final (ex: 3 para RGB).

    Retorna:
        List[torch.Tensor]: lista de tensores redimensionados para o tamanho da saída final.
    """
    outputs = []
    target_size = decoder_outputs[0].shape[2:]  # assume que o primeiro é a resolução final
    for i, d in enumerate(decoder_outputs):
        # Ajusta para a resolução final usando interpolação bilinear
        out = F.interpolate(d, size=target_size, mode='bilinear', align_corners=False)
        # Projeta para o número de canais da saída final
        if d.shape[1] != final_channels:
            proj = nn.Conv2d(d.shape[1], final_channels, kernel_size=1)
            out = proj(out)
        outputs.append(out)
    return outputs

def compute_deep_supervision_loss(predictions, target, criterion):
    """
    Computa a loss de deep supervision combinando perdas de múltiplos níveis.
    
    Args:
        predictions (list of torch.Tensor): lista de saídas intermediárias do decoder.
        target (torch.Tensor): imagem ground-truth.
        criterion (nn.Module): função de loss (ex: L1, MSE, etc.)
    
    Retorna:
        torch.Tensor: loss total somada de todos os níveis.
    """
    loss = 0.0
    for pred in predictions:
        loss += criterion(pred, target)
    return loss / len(predictions)  # média das perdas
