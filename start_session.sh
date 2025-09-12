#!/bin/bash

conda activate ImageStitching

# Nome da sessão
SESSION="gan_session"

# Cria nova sessão tmux em modo detached
tmux new-session -d -s $SESSION

# Painel 1 - roda o script de treino
tmux send-keys -t $SESSION "bash /home/prkd/gan/carregarTesteAtual.sh" C-m

# Painel 2 - divide horizontalmente e roda monitoramento GPU
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION:0.1 "watch -n 1 nvidia-smi" C-m

# Atacha na sessão
tmux attach -t $SESSION
