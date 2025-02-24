
# python finetune_lbnm.py --few_shot 0.01 --datanames hcpya --pretrained_datanames  adni ppmi abide taowu neurocon --load_dname ppmi --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot001_hcpya_5datasetDecoder32_attrFC.log & 
# pid1=$!
# python finetune_lbnm.py --few_shot 0.01 --datanames hcpa --pretrained_datanames  adni ppmi abide taowu neurocon --load_dname ppmi --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot001_hcpa_5datasetDecoder32_attrFC.log & 
# pid2=$!
# python finetune_lbnm.py --few_shot 0.01 --datanames adni --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot001_adni_6datasetDecoder32_attrFC.log & 
# pid3=$!
# wait $pid1
# wait $pid2
# wait $pid3
# python finetune_lbnm.py --few_shot 0.01 --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot001_ppmi_6datasetDecoder32_attrFC.log & 
# pid4=$!
# python finetune_lbnm.py --few_shot 0.01 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot001_abide_6datasetDecoder32_attrFC.log & 
# pid5=$!
# python finetune_lbnm.py --few_shot 0.01 --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot001_neurocon_2datasetDecoder32_attrFC.log & 
# pid6=$!
# wait $pid4
# wait $pid5
# wait $pid6
# python finetune_lbnm.py --few_shot 0.01 --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot001_taowu_2datasetDecoder32_attrFC.log & 
# pid7=$!
# wait $pid7



# python finetune_lbnm.py --few_shot 0.1 --datanames hcpya --pretrained_datanames  adni ppmi abide taowu neurocon --load_dname ppmi --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot01_hcpya_5datasetDecoder32_attrFC.log & 
# pid1=$!
# python finetune_lbnm.py --few_shot 0.1 --datanames hcpa --pretrained_datanames  adni ppmi abide taowu neurocon --load_dname ppmi --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot01_hcpa_5datasetDecoder32_attrFC.log & 
# pid2=$!
# python finetune_lbnm.py --few_shot 0.1 --datanames adni --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot01_adni_6datasetDecoder32_attrFC.log & 
# pid3=$!
# wait $pid1
# wait $pid2
# wait $pid3
# python finetune_lbnm.py --few_shot 0.1 --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot01_ppmi_6datasetDecoder32_attrFC.log & 
# pid4=$!
# python finetune_lbnm.py --few_shot 0.1 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot01_abide_6datasetDecoder32_attrFC.log & 
# pid5=$!
# python finetune_lbnm.py --few_shot 0.1 --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot01_neurocon_2datasetDecoder32_attrFC.log & 
# pid6=$!
# wait $pid4
# wait $pid5
# wait $pid6
# python finetune_lbnm.py --few_shot 0.1 --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot01_taowu_2datasetDecoder32_attrFC.log & 
# pid7=$!
# wait $pid7



# python finetune_lbnm.py --few_shot 0.3 --datanames hcpya --pretrained_datanames  adni ppmi abide taowu neurocon --load_dname ppmi --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot03_hcpya_5datasetDecoder32_attrFC.log & 
# pid1=$!
# python finetune_lbnm.py --few_shot 0.3 --datanames hcpa --pretrained_datanames  adni ppmi abide taowu neurocon --load_dname ppmi --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot03_hcpa_5datasetDecoder32_attrFC.log & 
# pid2=$!
# python finetune_lbnm.py --few_shot 0.3 --datanames adni --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot03_adni_6datasetDecoder32_attrFC.log & 
# pid3=$!
# wait $pid1
# wait $pid2
# wait $pid3
python finetune_lbnm.py --few_shot 0.3 --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot03_ppmi_6datasetDecoder32_attrFC.log & 
# pid4=$!
# python finetune_lbnm.py --few_shot 0.3 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot03_abide_6datasetDecoder32_attrFC.log & 
# pid5=$!
# python finetune_lbnm.py --few_shot 0.3 --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot03_neurocon_2datasetDecoder32_attrFC.log & 
# pid6=$!
# wait $pid4
# wait $pid5
# wait $pid6
# python finetune_lbnm.py --few_shot 0.3 --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot03_taowu_2datasetDecoder32_attrFC.log & 
# pid7=$!
# wait $pid7



# python finetune_lbnm.py --few_shot 0.5 --datanames hcpya --pretrained_datanames  adni ppmi abide taowu neurocon --load_dname ppmi --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot05_hcpya_5datasetDecoder32_attrFC.log & 
# pid1=$!
# python finetune_lbnm.py --few_shot 0.5 --datanames hcpa --pretrained_datanames  adni ppmi abide taowu neurocon --load_dname ppmi --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot05_hcpa_5datasetDecoder32_attrFC.log & 
# pid2=$!
# python finetune_lbnm.py --few_shot 0.5 --datanames adni --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot05_adni_6datasetDecoder32_attrFC.log & 
# pid3=$!
# wait $pid1
# wait $pid2
# wait $pid3
python finetune_lbnm.py --few_shot 0.5 --datanames ppmi --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot05_ppmi_6datasetDecoder32_attrFC.log & 
# pid4=$!
# python finetune_lbnm.py --few_shot 0.5 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot05_abide_6datasetDecoder32_attrFC.log & 
# pid5=$!
# python finetune_lbnm.py --few_shot 0.5 --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot05_neurocon_2datasetDecoder32_attrFC.log & 
# pid6=$!
# wait $pid4
# wait $pid5
# wait $pid6
# python finetune_lbnm.py --few_shot 0.5 --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShot05_taowu_2datasetDecoder32_attrFC.log & 
# pid7=$!
# wait $pid7

python finetune_lbnm.py --few_shot 0.01 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv3001_ppmi2cls_6datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 0.1 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv301_ppmi2cls_6datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 0.3 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv303_ppmi2cls_6datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 0.5 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv305_ppmi2cls_6datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 1 --datanames ppmi --force_2class --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:2 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv310_ppmi2cls_6datasetDecoder32_attrFC.log 



# python finetune_lbnm.py --few_shot 0.01 --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv2001_neurocon_2datasetDecoder32_attrFC.log &
# python finetune_lbnm.py --few_shot 0.01 --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv2001_taowu_2datasetDecoder32_attrFC.log 
# python finetune_lbnm.py --few_shot 0.1 --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv201_neurocon_2datasetDecoder32_attrFC.log &
# python finetune_lbnm.py --few_shot 0.1 --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv201_taowu_2datasetDecoder32_attrFC.log 
# python finetune_lbnm.py --few_shot 0.3 --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv203_neurocon_2datasetDecoder32_attrFC.log &
# python finetune_lbnm.py --few_shot 0.3 --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv203_taowu_2datasetDecoder32_attrFC.log 
# python finetune_lbnm.py --few_shot 0.5 --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv205_neurocon_2datasetDecoder32_attrFC.log &
# python finetune_lbnm.py --few_shot 0.5 --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv205_taowu_2datasetDecoder32_attrFC.log 

# python finetune_lbnm.py --few_shot 0.01 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv2001_abide_6datasetDecoder32_attrFC.log 
# python finetune_lbnm.py --few_shot 0.1 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv201_abide_6datasetDecoder32_attrFC.log 
# python finetune_lbnm.py --few_shot 0.3 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv203_abide_6datasetDecoder32_attrFC.log 
# python finetune_lbnm.py --few_shot 0.5 --datanames abide --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:4 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv205_abide_6datasetDecoder32_attrFC.log 


python finetune_lbnm.py --few_shot 0.01 --datanames neurocon  --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv2001_neurocon_6datasetDecoder32_attrFC.log &
python finetune_lbnm.py --few_shot 0.01 --datanames taowu  --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv2001_taowu_6datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 0.1 --datanames neurocon  --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv201_neurocon_6datasetDecoder32_attrFC.log &
python finetune_lbnm.py --few_shot 0.1 --datanames taowu  --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv201_taowu_6datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 0.3 --datanames neurocon  --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv203_neurocon_6datasetDecoder32_attrFC.log &
python finetune_lbnm.py --few_shot 0.3 --datanames taowu  --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv203_taowu_6datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 0.5 --datanames neurocon  --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:0 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv205_neurocon_6datasetDecoder32_attrFC.log &
python finetune_lbnm.py --few_shot 0.5 --datanames taowu  --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:1 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv205_taowu_6datasetDecoder32_attrFC.log 

python finetune_lbnm.py --few_shot 0.01 --datanames abide --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv3001_abide_7datasetDecoder32_attrFC.log  &
python finetune_lbnm.py --few_shot 0.1 --datanames abide --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv301_abide_7datasetDecoder32_attrFC.log  
python finetune_lbnm.py --few_shot 0.3 --datanames abide --device cuda:3 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv303_abide_7datasetDecoder32_attrFC.log  &
python finetune_lbnm.py --few_shot 0.5 --datanames abide --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > logs3/noneFewShotv305_abide_7datasetDecoder32_attrFC.log 


python finetune_lbnm.py --few_shot 0.01 --datanames sz-diana --device cuda:0 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv2001_sz-diana_2datasetDecoder32_attrFC.log  
python finetune_lbnm.py --few_shot 0.1 --datanames sz-diana --device cuda:1 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv201_sz-diana_2datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 0.3 --datanames sz-diana --device cuda:0 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv203_sz-diana_2datasetDecoder32_attrFC.log  &
python finetune_lbnm.py --few_shot 0.5 --datanames sz-diana --device cuda:1 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv205_sz-diana_2datasetDecoder32_attrFC.log 
python finetune_lbnm.py --few_shot 1 --datanames sz-diana --device cuda:0 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv210_sz-diana_2datasetDecoder32_attrFC.log 

python finetune_lbnm.py --few_shot 0.1 --datanames abide --device cuda:0 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv301_abide_2datasetDecoder32_attrFC.log  
python finetune_lbnm.py --few_shot 0.01 --datanames abide --device cuda:1 --decoder --decoder_layer 32 --pretrained_datanames hcpa hcpya > logs3/noneFewShotv3001_abide_2datasetDecoder32_attrFC.log  
