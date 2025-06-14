#!/usr/bin/env bash
set -e

# Default Training Parameters
DATA_ROOTPATH="E:/MDPP_data/MPDD-Elderly"
TRAIN_MODEL="D:/HACI/MMchallenge/MEIJU2025-baseline-master/MPDD/checkpoints/1s_5labels_opensmile+densenet/best_model_2025-02-13-21.12.01.pth"
AUDIOFEATURE_METHOD="opensmile" # Audio feature type, options {wav2vec, opensmile, mfccs}
VIDEOLFEATURE_METHOD="densenet" # Video feature type, options {openface, resnet, densenet}
SPLITWINDOW="1s" # Window duration, options {"1s", "5s"}
LABELCOUNT=5 # Number of label categories, options {2, 3, 5}
TRACK_OPTION="Track1"
FEATURE_MAX_LEN=26 # Set maximum feature length; pad with zeros if insufficient, truncate if exceeding. For Track1, options {26, 5}; for Track2, options {25, 5}
BATCH_SIZE=1
DEVICE="cpu"

for arg in "$@"; do
  case $arg in
    --data_rootpath=*) DATA_ROOTPATH="${arg#*=}" ;;
    --train_model=*) TRAIN_MODEL="${arg#*=}" ;;
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
    cmd="python test.py \
        --data_rootpath=$DATA_ROOTPATH \
        --train_model=$TRAIN_MODEL \
        --audiofeature_method=$AUDIOFEATURE_METHOD \
        --videofeature_method=$VIDEOLFEATURE_METHOD \
        --splitwindow_time=$SPLITWINDOW \
        --labelcount=$LABELCOUNT \
        --track_option=$TRACK_OPTION \
        --feature_max_len=$FEATURE_MAX_LEN \
        --batch_size=$BATCH_SIZE \
        --device=$DEVICE"

    echo "\n-------------------------------------------------------------------------------------"
    echo "Execute command: $cmd"
    echo "-------------------------------------------------------------------------------------\n"
    echo $cmd | sh
done
