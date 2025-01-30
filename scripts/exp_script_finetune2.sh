python finetune_lbnm.py --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32NoDis_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32NoDis_attrFC.log 
python finetune_lbnm.py --datanames taowu --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32NoAD_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32NoAD_attrFC.log 
python finetune_lbnm.py --datanames taowu --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32NoPD_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32NoPD_attrFC.log 
python finetune_lbnm.py --datanames taowu --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32NoAt_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32NoAt_attrFC.log 
python finetune_lbnm.py --datanames taowu --pretrained_datanames ppmi abide taowu neurocon hcpa adni hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32Full_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames ppmi abide taowu neurocon hcpa adni hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32Full_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32NoDis_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32NoAD_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32NoPD_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32NoAt_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames ppmi abide taowu neurocon hcpa adni hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32Full_attrFC.log 
