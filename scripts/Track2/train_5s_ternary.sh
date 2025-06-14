#!/usr/bin/env bash
set -e

# Default Training Parameters
data_rootpath="D:/HACI/MMchallenge/NEUQdata" # Dataset root directory
AUDIOFEATURE_METHOD="mfccs" # Audio feature type, options {wav2vec, opensmile, mfccs}
VIDEOLFEATURE_METHOD="densenet" # Video feature type, options {openface, resnet, densenet}
SPLITWINDOW="5s" # Window duration, options {"1s", "5s"}
LABELCOUNT=3 # Number of label categories, options {2, 3, 5}
TRACK_OPTION="Track2"
FEATURE_MAX_LEN=5 # Set maximum feature length; pad with zeros if insufficient, truncate if exceeding
BATCH_SIZE=8
LR=0.0004
NUM_EPOCHS=500
DEVICE="cpu" # Options {cuda, cpu}


for arg in "$@"; do
  case $arg in
    --data_rootpath=*) data_rootpath="${arg#*=}" ;;
    --audiofeature_method=*) AUDIOFEATURE_METHOD="${arg#*=}" ;;
    --videofeature_method=*) VIDEOLFEATURE_METHOD="${arg#*=}" ;;
    --splitwindow_time=*) SPLITWINDOW="${arg#*=}" ;;
    --labelcount=*) LABELCOUNT="${arg#*=}" ;;
    --track_option=*) TRACK_OPTION="${arg#*=}" ;;
    --feature_max_len=*) FEATURE_MAX_LEN="${arg#*=}" ;;
    --batch_size=*) BATCH_SIZE="${arg#*=}" ;;
    --lr=*) LR="${arg#*=}" ;;
    --num_epochs=*) NUM_EPOCHS="${arg#*=}" ;;
    --device=*) DEVICE="${arg#*=}" ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

for i in `seq 1 1 1`; do
    cmd="python train.py \
        --data_rootpath=$data_rootpath \
        --audiofeature_method=$AUDIOFEATURE_METHOD \
        --videofeature_method=$VIDEOLFEATURE_METHOD \
        --splitwindow_time=$SPLITWINDOW \
        --labelcount=$LABELCOUNT \
        --track_option=$TRACK_OPTION \
        --feature_max_len=$FEATURE_MAX_LEN \
        --batch_size=$BATCH_SIZE \
        --lr=$LR \
        --num_epochs=$NUM_EPOCHS \
        --device=$DEVICE"

    echo "\n-------------------------------------------------------------------------------------"
    echo "Execute command: $cmd"
    echo "-------------------------------------------------------------------------------------\n"
    echo $cmd | sh
done