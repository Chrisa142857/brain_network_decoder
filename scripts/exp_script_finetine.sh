

# python finetune_lbnm.py --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > none_5datanames_decoder32NoAD_attrFC.log


# python finetune_lbnm.py --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > none_5datanames_decoder32NoPD_attrFC.log


# python finetune_lbnm.py --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > none_5datanames_decoder32NoAt_attrFC.log


# python finetune_lbnm.py --pretrained_datanames hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_5datanames_decoder32NoDis_attrFC.log

python finetune_lbnm.py --datanames ppmi --pretrained_datanames hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > none_ppmi_decoder32NoDis_attrFC.log &
pid1=$!
python finetune_lbnm.py --datanames abide --pretrained_datanames hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > none_abide_decoder32NoDis_attrFC.log &
pid2=$!
python finetune_lbnm.py --datanames ppmi --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > none_ppmi_decoder32NoAD_attrFC.log &
pid3=$!
python finetune_lbnm.py --datanames abide --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > none_abide_decoder32NoAD_attrFC.log &
pid4=$!
python finetune_lbnm.py --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > none_ppmi_decoder32NoPD_attrFC.log &
pid5=$!
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5

python finetune_lbnm.py --datanames abide --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > none_abide_decoder32NoPD_attrFC.log &
pid1=$!
python finetune_lbnm.py --datanames ppmi --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > none_ppmi_decoder32NoAt_attrFC.log &
pid2=$!
python finetune_lbnm.py --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > none_abide_decoder32NoAt_attrFC.log &
pid3=$!
python finetune_lbnm.py --datanames ppmi --pretrained_datanames ppmi abide taowu neurocon hcpa adni hcpya --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > none_ppmi_decoder32Full_attrFC.log &
pid4=$!
python finetune_lbnm.py --datanames abide --pretrained_datanames ppmi abide taowu neurocon hcpa adni hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > none_abide_decoder32Full_attrFC.log &
pid5=$!
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5