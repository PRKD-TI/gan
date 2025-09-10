#!/bin/bash

# Link de compartilhamento do OneDrive (curto)
LINK="https://1drv.ms/u/c/af1da744f6d314ca/Ec6-z9CsPmBMvQFdEMjZtTkBkT0ZsU6mIC-ClEHfxvypGg?e=ablq4b"

# Codifica em base64 para API do OneDrive
ENC=$(echo -n "$LINK" | base64 | tr '/+' '_-' | tr -d '=')

# Monta a URL final para download direto
URL="https://api.onedrive.com/v1.0/shares/u!${ENC}/root/content"

# Nome do arquivo de saída
OUTPUT="512x256.zip"

echo "Baixando arquivo do OneDrive..."
echo "URL convertida: $URL"
echo "Saída: $OUTPUT"
echo

# Baixar com barra de progresso
curl -L "$URL" -o "$OUTPUT" --progress-bar

