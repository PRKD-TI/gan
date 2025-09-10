#!/bin/bash

# Caminhos
ZIP_RAM=/mnt/ramdisk/512x256.zip
OUT_RAM=/mnt/ramdiskdataset

# Criar diretório de destino se não existir
mkdir -p "$OUT_RAM"

# Extrair usando 35 núcleos, sem exibir cada arquivo
unzip -Z1 "$ZIP_RAM" | \
xargs -n1 -P35 -I{} unzip -qq "$ZIP_RAM" "{}" -d "$OUT_RAM"
