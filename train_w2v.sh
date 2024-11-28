#!/bin/bash

# Parâmetros que serão usados
VOCAB_SIZES=(50 100)
WINDOW_SIZES=(5 10)
EPOCHS=(100 250)
VECTOR_LENGTHS=(250)

# Caminho do arquivo de treino e o caminho para salvar o modelo
TRAIN_PATH=$1
SAVE_PATH=$2

# Função para calcular o tempo e executar o comando Python
train_model() {
    local vocab_size=$1
    local window_size=$2
    local num_epochs=$3
    local vector_length=$4
    local save_path=$5
    
    # Cria a pasta save_path se não existir
    mkdir -p "$(dirname "$save_path")"

    # Caminho do log
    LOG_FILE="${save_path}_log.txt"

    echo "Treinando modelo..."
    START_TIME=$(date +%s)

    # Executa o script Python com os parâmetros
    python ~/ufpr-thesis/w2v.py -tp "$TRAIN_PATH" -vs "$vocab_size" -ws "$window_size" -ne "$num_epochs" -op "$save_path" -vl "$vector_length"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # Salva o tempo de execução no arquivo de log
    echo "Tempo para vocab_size=$vocab_size, window_size=$window_size, epochs=$num_epochs, vector_length=$vector_length: $DURATION segundos" | tee -a $LOG_FILE
}

# Loop pelas combinações de parâmetros
for vocab_size in "${VOCAB_SIZES[@]}"; do
    for window_size in "${WINDOW_SIZES[@]}"; do
        for num_epoch in "${EPOCHS[@]}"; do
            for vector_length in "${VECTOR_LENGTHS[@]}"; do
                SAVE_MODEL_PATH="${SAVE_PATH}/${vocab_size}_${window_size}_${num_epoch}_${vector_length}_model"
                train_model $vocab_size $window_size $num_epoch $vector_length $SAVE_MODEL_PATH
            done
        done
    done
done
