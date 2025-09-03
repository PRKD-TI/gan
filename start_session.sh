#!/bin/bash

# Nome da sessão
SESSION="gan_session"

# Cria nova sessão tmux em modo detached
tmux new-session -d -s $SESSION

# Painel 1 - roda o script de treino
tmux send-keys -t $SESSION "bash /home/training/gan/carregarTesteAtual.sh" C-m

# Painel 2 - divide horizontalmente e roda monitoramento GPU
tmux split-window -h -t $SESSION
tmux send-keys -t $SESSION:0.1 "watch -n 1 nvidia-smi" C-m

# Painel 3 - divide o painel da esquerda em duas linhas e roda tensorboard
tmux select-pane -t $SESSION:0.1
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION:0.2 "tensorboard --logdir=./logs --port=8080 --host=0.0.0.0" C-m

# Atacha na sessão
tmux attach -t $SESSION
