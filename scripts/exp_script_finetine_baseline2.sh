
# # python trainval.py --few_shot 0.01 --models graphormer --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot001_hcpya_mlp1_attrFC.log 
# # pid1=$!
# # python trainval.py --few_shot 0.01 --models graphormer --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot001_hcpa_mlp1_attrFC.log 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.01 --models graphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot001_adni_mlp1_attrFC.log 
# pid3=$!
# python trainval.py --few_shot 0.01 --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot001_ppmi_mlp1_attrFC.log 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.01 --models graphormer --dataname abide --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot001_abide_mlp1_attrFC.log 
# pid5=$!
# python trainval.py --few_shot 0.01 --models graphormer --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot001_neurocon_mlp1_attrFC.log 
# pid6=$!
# python trainval.py --few_shot 0.01 --models graphormer --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot001_taowu_mlp1_attrFC.log 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7


# # python trainval.py --few_shot 0.1 --models graphormer --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot01_hcpya_mlp1_attrFC.log 
# # pid1=$!
# # python trainval.py --few_shot 0.1 --models graphormer --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot01_hcpa_mlp1_attrFC.log 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.1 --models graphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot01_adni_mlp1_attrFC.log 
# pid3=$!
# python trainval.py --few_shot 0.1 --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot01_ppmi_mlp1_attrFC.log 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.1 --models graphormer --dataname abide --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot01_abide_mlp1_attrFC.log 
# pid5=$!
# python trainval.py --few_shot 0.1 --models graphormer --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot01_neurocon_mlp1_attrFC.log 
# pid6=$!
# python trainval.py --few_shot 0.1 --models graphormer --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot01_taowu_mlp1_attrFC.log 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7




# # python trainval.py --few_shot 0.3 --models graphormer --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot03_hcpya_mlp1_attrFC.log 
# # pid1=$!
# # python trainval.py --few_shot 0.3 --models graphormer --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot03_hcpa_mlp1_attrFC.log 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.3 --models graphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot03_adni_mlp1_attrFC.log 
# pid3=$!
# python trainval.py --few_shot 0.3 --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot03_ppmi_mlp1_attrFC.log 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.3 --models graphormer --dataname abide --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot03_abide_mlp1_attrFC.log 
# pid5=$!
# python trainval.py --few_shot 0.3 --models graphormer --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot03_neurocon_mlp1_attrFC.log 
# pid6=$!
# python trainval.py --few_shot 0.3 --models graphormer --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot03_taowu_mlp1_attrFC.log 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7



# # python trainval.py --few_shot 0.5 --models graphormer --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot05_hcpya_mlp1_attrFC.log 
# # pid1=$!
# # python trainval.py --few_shot 0.5 --models graphormer --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot05_hcpa_mlp1_attrFC.log 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.5 --models graphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot05_adni_mlp1_attrFC.log 
# pid3=$!
# python trainval.py --few_shot 0.5 --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot05_ppmi_mlp1_attrFC.log 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.5 --models graphormer --dataname abide --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot05_abide_mlp1_attrFC.log 
# pid5=$!
# python trainval.py --few_shot 0.5 --models graphormer --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot05_neurocon_mlp1_attrFC.log 
# pid6=$!
# python trainval.py --few_shot 0.5 --models graphormer --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/graphormerFewShot05_taowu_mlp1_attrFC.log 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7




# python trainval.py --few_shot 0.01 --models nagphormer --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot001_hcpya_mlp1_attrFC.log 
# pid1=$!
# python trainval.py --few_shot 0.01 --models nagphormer --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot001_hcpa_mlp1_attrFC.log 
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.01 --models nagphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot001_adni_mlp1_attrFC.log 
pid3=$!
python trainval.py --few_shot 0.01 --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot001_ppmi_mlp1_attrFC.log 
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.01 --models nagphormer --dataname abide --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot001_abide_mlp1_attrFC.log 
pid5=$!
python trainval.py --few_shot 0.01 --models nagphormer --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot001_neurocon_mlp1_attrFC.log 
pid6=$!
python trainval.py --few_shot 0.01 --models nagphormer --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot001_taowu_mlp1_attrFC.log 
pid7=$!
wait $pid5
wait $pid6
wait $pid7


# python trainval.py --few_shot 0.1 --models nagphormer --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot01_hcpya_mlp1_attrFC.log 
# pid1=$!
# python trainval.py --few_shot 0.1 --models nagphormer --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot01_hcpa_mlp1_attrFC.log 
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.1 --models nagphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot01_adni_mlp1_attrFC.log 
pid3=$!
python trainval.py --few_shot 0.1 --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot01_ppmi_mlp1_attrFC.log 
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.1 --models nagphormer --dataname abide --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot01_abide_mlp1_attrFC.log 
pid5=$!
python trainval.py --few_shot 0.1 --models nagphormer --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot01_neurocon_mlp1_attrFC.log 
pid6=$!
python trainval.py --few_shot 0.1 --models nagphormer --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot01_taowu_mlp1_attrFC.log 
pid7=$!
wait $pid5
wait $pid6
wait $pid7




# python trainval.py --few_shot 0.3 --models nagphormer --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot03_hcpya_mlp1_attrFC.log 
# pid1=$!
# python trainval.py --few_shot 0.3 --models nagphormer --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot03_hcpa_mlp1_attrFC.log 
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.3 --models nagphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot03_adni_mlp1_attrFC.log 
pid3=$!
python trainval.py --few_shot 0.3 --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot03_ppmi_mlp1_attrFC.log 
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.3 --models nagphormer --dataname abide --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot03_abide_mlp1_attrFC.log 
pid5=$!
python trainval.py --few_shot 0.3 --models nagphormer --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot03_neurocon_mlp1_attrFC.log 
pid6=$!
python trainval.py --few_shot 0.3 --models nagphormer --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot03_taowu_mlp1_attrFC.log 
pid7=$!
wait $pid5
wait $pid6
wait $pid7



# python trainval.py --few_shot 0.5 --models nagphormer --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot05_hcpya_mlp1_attrFC.log 
# pid1=$!
# python trainval.py --few_shot 0.5 --models nagphormer --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot05_hcpa_mlp1_attrFC.log 
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.5 --models nagphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot05_adni_mlp1_attrFC.log 
pid3=$!
python trainval.py --few_shot 0.5 --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot05_ppmi_mlp1_attrFC.log 
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.5 --models nagphormer --dataname abide --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot05_abide_mlp1_attrFC.log 
pid5=$!
python trainval.py --few_shot 0.5 --models nagphormer --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot05_neurocon_mlp1_attrFC.log 
pid6=$!
python trainval.py --few_shot 0.5 --models nagphormer --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/nagphormerFewShot05_taowu_mlp1_attrFC.log 
pid7=$!
wait $pid5
wait $pid6
wait $pid7




# python trainval.py --few_shot 0.01 --models braingnn --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot001_hcpya_mlp1_attrFC.log &
# pid1=$!
# python trainval.py --few_shot 0.01 --models braingnn --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot001_hcpa_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.01 --models braingnn --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot001_adni_mlp1_attrFC.log &
pid3=$!
python trainval.py --few_shot 0.01 --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot001_ppmi_mlp1_attrFC.log &
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.01 --models braingnn --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot001_abide_mlp1_attrFC.log &
pid5=$!
python trainval.py --few_shot 0.01 --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot001_neurocon_mlp1_attrFC.log &
pid6=$!
python trainval.py --few_shot 0.01 --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot001_taowu_mlp1_attrFC.log &
pid7=$!
wait $pid5
wait $pid6
wait $pid7


# python trainval.py --few_shot 0.1 --models braingnn --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot01_hcpya_mlp1_attrFC.log &
# pid1=$!
# python trainval.py --few_shot 0.1 --models braingnn --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot01_hcpa_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.1 --models braingnn --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot01_adni_mlp1_attrFC.log &
pid3=$!
python trainval.py --few_shot 0.1 --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot01_ppmi_mlp1_attrFC.log &
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.1 --models braingnn --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot01_abide_mlp1_attrFC.log &
pid5=$!
python trainval.py --few_shot 0.1 --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot01_neurocon_mlp1_attrFC.log &
pid6=$!
python trainval.py --few_shot 0.1 --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot01_taowu_mlp1_attrFC.log &
pid7=$!
wait $pid5
wait $pid6
wait $pid7




# python trainval.py --few_shot 0.3 --models braingnn --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot03_hcpya_mlp1_attrFC.log &
# pid1=$!
# python trainval.py --few_shot 0.3 --models braingnn --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot03_hcpa_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.3 --models braingnn --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot03_adni_mlp1_attrFC.log &
pid3=$!
python trainval.py --few_shot 0.3 --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot03_ppmi_mlp1_attrFC.log &
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.3 --models braingnn --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot03_abide_mlp1_attrFC.log &
pid5=$!
python trainval.py --few_shot 0.3 --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot03_neurocon_mlp1_attrFC.log &
pid6=$!
python trainval.py --few_shot 0.3 --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot03_taowu_mlp1_attrFC.log &
pid7=$!
wait $pid5
wait $pid6
wait $pid7



# python trainval.py --few_shot 0.5 --models braingnn --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot05_hcpya_mlp1_attrFC.log &
# pid1=$!
# python trainval.py --few_shot 0.5 --models braingnn --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot05_hcpa_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.5 --models braingnn --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot05_adni_mlp1_attrFC.log &
pid3=$!
python trainval.py --few_shot 0.5 --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot05_ppmi_mlp1_attrFC.log &
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.5 --models braingnn --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot05_abide_mlp1_attrFC.log &
pid5=$!
python trainval.py --few_shot 0.5 --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot05_neurocon_mlp1_attrFC.log &
pid6=$!
python trainval.py --few_shot 0.5 --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot05_taowu_mlp1_attrFC.log &
pid7=$!
wait $pid5
wait $pid6
wait $pid7




# python trainval.py --few_shot 0.01 --models neurodetour --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_hcpya_mlp1_attrFC.log &
# pid1=$!
# python trainval.py --few_shot 0.01 --models neurodetour --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_hcpa_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.01 --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_adni_mlp1_attrFC.log &
pid3=$!
python trainval.py --few_shot 0.01 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_ppmi_mlp1_attrFC.log &
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.01 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_abide_mlp1_attrFC.log &
pid5=$!
python trainval.py --few_shot 0.01 --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_neurocon_mlp1_attrFC.log &
pid6=$!
python trainval.py --few_shot 0.01 --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_taowu_mlp1_attrFC.log &
pid7=$!
wait $pid5
wait $pid6
wait $pid7


# python trainval.py --few_shot 0.1 --models neurodetour --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_hcpya_mlp1_attrFC.log &
# pid1=$!
# python trainval.py --few_shot 0.1 --models neurodetour --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_hcpa_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.1 --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_adni_mlp1_attrFC.log &
pid3=$!
python trainval.py --few_shot 0.1 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_ppmi_mlp1_attrFC.log &
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.1 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_abide_mlp1_attrFC.log &
pid5=$!
python trainval.py --few_shot 0.1 --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_neurocon_mlp1_attrFC.log &
pid6=$!
python trainval.py --few_shot 0.1 --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_taowu_mlp1_attrFC.log &
pid7=$!
wait $pid5
wait $pid6
wait $pid7




# python trainval.py --few_shot 0.3 --models neurodetour --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_hcpya_mlp1_attrFC.log &
# pid1=$!
# python trainval.py --few_shot 0.3 --models neurodetour --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_hcpa_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.3 --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_adni_mlp1_attrFC.log &
pid3=$!
python trainval.py --few_shot 0.3 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_ppmi_mlp1_attrFC.log &
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.3 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_abide_mlp1_attrFC.log &
pid5=$!
python trainval.py --few_shot 0.3 --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_neurocon_mlp1_attrFC.log &
pid6=$!
python trainval.py --few_shot 0.3 --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_taowu_mlp1_attrFC.log &
pid7=$!
wait $pid5
wait $pid6
wait $pid7



# python trainval.py --few_shot 0.5 --models neurodetour --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_hcpya_mlp1_attrFC.log &
# pid1=$!
# python trainval.py --few_shot 0.5 --models neurodetour --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_hcpa_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2
python trainval.py --few_shot 0.5 --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_adni_mlp1_attrFC.log &
pid3=$!
python trainval.py --few_shot 0.5 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_ppmi_mlp1_attrFC.log &
pid4=$!
wait $pid3
wait $pid4
python trainval.py --few_shot 0.5 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_abide_mlp1_attrFC.log &
pid5=$!
python trainval.py --few_shot 0.5 --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_neurocon_mlp1_attrFC.log &
pid6=$!
python trainval.py --few_shot 0.5 --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_taowu_mlp1_attrFC.log &
pid7=$!
wait $pid5
wait $pid6
wait $pid7



