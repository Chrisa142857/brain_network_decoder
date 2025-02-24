
# python trainval.py --few_shot 0.01 --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot001_hcpya_ranDecoder32_attrFC.log & 
# pid1=$!
# python trainval.py --few_shot 0.01 --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot001_hcpa_ranDecoder32_attrFC.log & 
# pid2=$!
# wait $pid1
# wait $pid2
# python trainval.py --few_shot 0.01 --models none --dataname adni --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot001_adni_ranDecoder32_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.01 --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot001_ppmi_ranDecoder32_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.01 --models none --dataname abide --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot001_abide_ranDecoder32_attrFC.log & 
# pid5=$!
python trainval.py --few_shot 0.01 --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot001_neurocon_ranDecoder32_attrFC.log 
# pid6=$!
# python trainval.py --few_shot 0.01 --models none --dataname taowu --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot001_taowu_ranDecoder32_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7


# python trainval.py --few_shot 0.1 --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot01_hcpya_ranDecoder32_attrFC.log & 
# pid1=$!
# python trainval.py --few_shot 0.1 --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot01_hcpa_ranDecoder32_attrFC.log & 
# pid2=$!
# wait $pid1
# wait $pid2
# python trainval.py --few_shot 0.1 --models none --dataname adni --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot01_adni_ranDecoder32_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.1 --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot01_ppmi_ranDecoder32_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.1 --models none --dataname abide --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot01_abide_ranDecoder32_attrFC.log & 
# pid5=$!
python trainval.py --few_shot 0.1 --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot01_neurocon_ranDecoder32_attrFC.log 
# pid6=$!
python trainval.py --few_shot 0.1 --models none --dataname taowu --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot01_taowu_ranDecoder32_attrFC.log 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7




# python trainval.py --few_shot 0.3 --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot03_hcpya_ranDecoder32_attrFC.log & 
# pid1=$!
# python trainval.py --few_shot 0.3 --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot03_hcpa_ranDecoder32_attrFC.log & 
# pid2=$!
# wait $pid1
# wait $pid2
# python trainval.py --few_shot 0.3 --models none --dataname adni --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot03_adni_ranDecoder32_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.3 --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot03_ppmi_ranDecoder32_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.3 --models none --dataname abide --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot03_abide_ranDecoder32_attrFC.log & 
# pid5=$!
python trainval.py --few_shot 0.3 --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot03_neurocon_ranDecoder32_attrFC.log 
# pid6=$!
# python trainval.py --few_shot 0.3 --models none --dataname taowu --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot03_taowu_ranDecoder32_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7



# python trainval.py --few_shot 0.5 --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot05_hcpya_ranDecoder32_attrFC.log & 
# pid1=$!
# python trainval.py --few_shot 0.5 --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot05_hcpa_ranDecoder32_attrFC.log & 
# pid2=$!
# wait $pid1
# wait $pid2
# python trainval.py --few_shot 0.5 --models none --dataname adni --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 800 > logs3/noneFewShot05_adni_ranDecoder32_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.5 --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot05_ppmi_ranDecoder32_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
python trainval.py --few_shot 0.5 --models none --dataname abide --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/noneFewShot05_abide_ranDecoder32_attrFC.log 
# pid5=$!
# python trainval.py --few_shot 0.5 --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot05_neurocon_ranDecoder32_attrFC.log & 
# pid6=$!
python trainval.py --few_shot 0.5 --models none --dataname taowu --node_attr FC --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 > logs3/noneFewShot05_taowu_ranDecoder32_attrFC.log 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7