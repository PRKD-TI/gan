# ------------------ checkpoint utils ------------------
import os
import re

def carregar_checkpoint_mais_recente(checkpoints_epoch_dir, checkpoints_batch_dir):
    pattern_epoch = re.compile(r"checkpoint_epoch(\d+)\.pt")
    pattern_batch = re.compile(r"checkpoint_epoch(\d+)_batch(\d+)\.pt")
    def extract_epoch_batch(filename):
        match = pattern_batch.match(filename)
        if match:
            return int(match.group(1)), int(match.group(2)), filename
        match = pattern_epoch.match(filename)
        if match:
            return int(match.group(1)), -1, filename
        return None
    all_checkpoints = []
    for fname in os.listdir(checkpoints_epoch_dir):
        result = extract_epoch_batch(fname)
        if result:
            epoch, batch, name = result
            all_checkpoints.append((epoch, batch, os.path.join(checkpoints_epoch_dir, name)))
    for fname in os.listdir(checkpoints_batch_dir):
        result = extract_epoch_batch(fname)
        if result:
            epoch, batch, name = result
            all_checkpoints.append((epoch, batch, os.path.join(checkpoints_batch_dir, name)))
    if not all_checkpoints:
        return None, 0, -1
    all_checkpoints.sort(key=lambda x: (x[0], x[1]))
    epoch, batch, path = all_checkpoints[-1]
    return path, epoch, batch
