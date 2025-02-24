
# python finetune_lbnm.py --few_shot 0.01 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv3001_ppmi2cls_6datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.01 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv3001_abide_6datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.1 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv301_ppmi2cls_6datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.1 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv301_abide_6datasetDecoder32_attrFC.log


# python finetune_lbnm.py --few_shot 0.3 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv303_ppmi2cls_6datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.3 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv303_abide_6datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.5 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv305_ppmi2cls_6datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.5 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv305_abide_6datasetDecoder32_attrFC.log


# python finetune_lbnm.py --few_shot 1 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv310_ppmi2cls_6datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 1 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv310_abide_6datasetDecoder32_attrFC.log 

# python finetune_lbnm.py --few_shot 0.01 --datanames sz-diana --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv3001_sz-diana_7datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.1 --datanames sz-diana --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv301_sz-diana_7datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.3 --datanames sz-diana --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv303_sz-diana_7datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.5 --datanames sz-diana --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv305_sz-diana_7datasetDecoder32_attrFC.log 

# python finetune_lbnm.py --few_shot 1 --datanames sz-diana --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv310_sz-diana_7datasetDecoder32_attrFC.log 


# python finetune_lbnm.py --few_shot 0.01 --datanames ppmi --force_2class --pretrained_datanames hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --lr 0.0001 > logs3/noneFewShotv4001_ppmi2cls_2datasetDecoder32_attrFC.log &
# sleep 10
# python finetune_lbnm.py --few_shot 0.1 --datanames ppmi --force_2class --pretrained_datanames hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 > logs3/noneFewShotv301_ppmi2cls_2datasetDecoder32_attrFC.log 
# python finetune_lbnm.py --few_shot 0.3 --datanames ppmi --force_2class --pretrained_datanames hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 > logs3/noneFewShotv303_ppmi2cls_2datasetDecoder32_attrFC.log 
# sleep 10
# python finetune_lbnm.py --few_shot 0.5 --datanames ppmi --force_2class --pretrained_datanames hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 > logs3/noneFewShotv305_ppmi2cls_2datasetDecoder32_attrFC.log &
# python finetune_lbnm.py --few_shot 1 --datanames ppmi --force_2class --pretrained_datanames hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 > logs3/noneFewShotv310_ppmi2cls_2datasetDecoder32_attrFC.log &


python finetune_lbnm.py --few_shot 0.01 --datanames sz-diana --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv3001_sz-diana_7datasetDecoder32_attrFC.log
python finetune_lbnm.py --few_shot 0.3 --datanames sz-diana --device cuda:5 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv203_sz-diana_2datasetDecoder32_attrFC.log
python finetune_lbnm.py --few_shot 0.5 --datanames sz-diana --device cuda:5 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv205_sz-diana_2datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 1 --datanames sz-diana --device cuda:5 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv210_sz-diana_2datasetDecoder32_attrFC.log 
