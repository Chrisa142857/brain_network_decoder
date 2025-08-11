

python finetune_lbnm.py --few_shot 0.01 --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --force_2class --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv2001_ppmi2cls_6datasetDecoder32_attrFC.log & 
pid5=$!
python finetune_lbnm.py --few_shot 0.1 --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --force_2class --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv201_ppmi2cls_6datasetDecoder32_attrFC.log & 
pid6=$!
python finetune_lbnm.py --few_shot 0.3 --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --force_2class --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv203_ppmi2cls_6datasetDecoder32_attrFC.log & 
pid7=$!
wait $pid5
wait $pid6
wait $pid7


python finetune_lbnm.py --few_shot 0.5 --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --force_2class --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv205_ppmi2cls_6datasetDecoder32_attrFC.log & 
pid5=$!
python finetune_lbnm.py --few_shot 1 --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --force_2class --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv210_ppmi2cls_6datasetDecoder32_attrFC.log & 
pid6=$!
python trainval.py --few_shot 0.01 --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 32 --force_2class --device cuda:3 --cv_fold_n 5 > logs3/noneFewShotv2001_ppmi2cls_ranDecoder32_attrFC.log 
python trainval.py --few_shot 1 --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 32 --force_2class --device cuda:4 --cv_fold_n 5 > logs3/noneFewShotv210_ppmi2cls_ranDecoder32_attrFC.log & 
pid7=$!
wait $pid5
wait $pid6
wait $pid7

python trainval.py --few_shot 0.1 --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 32 --force_2class --device cuda:2 --cv_fold_n 5 > logs3/noneFewShotv201_ppmi2cls_ranDecoder32_attrFC.log & 
pid5=$!
python trainval.py --few_shot 0.3 --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 32 --force_2class --device cuda:4 --cv_fold_n 5 > logs3/noneFewShotv203_ppmi2cls_ranDecoder32_attrFC.log & 
pid6=$!
python trainval.py --few_shot 0.5 --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 32 --force_2class --device cuda:3 --cv_fold_n 5 > logs3/noneFewShotv205_ppmi2cls_ranDecoder32_attrFC.log & 
pid7=$!
wait $pid5
wait $pid6
wait $pid7
