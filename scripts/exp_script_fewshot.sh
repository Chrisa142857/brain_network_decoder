
# # python trainval.py --few_shot 0.01 --models bnt --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot001_hcpya_mlp1_attrFC.log & 
# # pid1=$!
# # python trainval.py --few_shot 0.01 --models bnt --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot001_hcpa_mlp1_attrFC.log & 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.01 --models bnt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot001_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.01 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot001_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.01 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot001_abide_mlp1_attrFC.log & 
# pid5=$!
# python trainval.py --few_shot 0.01 --models bnt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot001_neurocon_mlp1_attrFC.log & 
# pid6=$!
# python trainval.py --few_shot 0.01 --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot001_taowu_mlp1_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7


# # python trainval.py --few_shot 0.1 --models bnt --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot01_hcpya_mlp1_attrFC.log & 
# # pid1=$!
# # python trainval.py --few_shot 0.1 --models bnt --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot01_hcpa_mlp1_attrFC.log & 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.1 --models bnt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot01_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.1 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot01_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.1 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot01_abide_mlp1_attrFC.log & 
# pid5=$!
# python trainval.py --few_shot 0.1 --models bnt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot01_neurocon_mlp1_attrFC.log & 
# pid6=$!
# python trainval.py --few_shot 0.1 --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot01_taowu_mlp1_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7




# # python trainval.py --few_shot 0.3 --models bnt --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot03_hcpya_mlp1_attrFC.log & 
# # pid1=$!
# # python trainval.py --few_shot 0.3 --models bnt --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot03_hcpa_mlp1_attrFC.log & 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.3 --models bnt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot03_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.3 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot03_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.3 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot03_abide_mlp1_attrFC.log & 
# pid5=$!
# python trainval.py --few_shot 0.3 --models bnt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot03_neurocon_mlp1_attrFC.log & 
# pid6=$!
# python trainval.py --few_shot 0.3 --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot03_taowu_mlp1_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7



# # python trainval.py --few_shot 0.5 --models bnt --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot05_hcpya_mlp1_attrFC.log & 
# # pid1=$!
# # python trainval.py --few_shot 0.5 --models bnt --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot05_hcpa_mlp1_attrFC.log & 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.5 --models bnt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot05_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.5 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot05_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.5 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot05_abide_mlp1_attrFC.log & 
# pid5=$!
# python trainval.py --few_shot 0.5 --models bnt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot05_neurocon_mlp1_attrFC.log & 
# pid6=$!
# python trainval.py --few_shot 0.5 --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 > logs3/bntFewShot05_taowu_mlp1_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7




# # python trainval.py --few_shot 0.01 --models bolt --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot001_hcpya_mlp1_attrFC.log & 
# # pid1=$!
# # python trainval.py --few_shot 0.01 --models bolt --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot001_hcpa_mlp1_attrFC.log & 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.01 --models bolt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot001_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.01 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot001_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.01 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 > logs3/boltFewShot001_abide_mlp1_attrFC.log & 
# pid5=$!
# python trainval.py --few_shot 0.01 --models bolt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot001_neurocon_mlp1_attrFC.log & 
# pid6=$!
# python trainval.py --few_shot 0.01 --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot001_taowu_mlp1_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7


# # python trainval.py --few_shot 0.1 --models bolt --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot01_hcpya_mlp1_attrFC.log & 
# # pid1=$!
# # python trainval.py --few_shot 0.1 --models bolt --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot01_hcpa_mlp1_attrFC.log & 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.1 --models bolt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot01_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.1 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot01_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.1 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 > logs3/boltFewShot01_abide_mlp1_attrFC.log & 
# pid5=$!
# python trainval.py --few_shot 0.1 --models bolt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot01_neurocon_mlp1_attrFC.log & 
# pid6=$!
# python trainval.py --few_shot 0.1 --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot01_taowu_mlp1_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7




# # python trainval.py --few_shot 0.3 --models bolt --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot03_hcpya_mlp1_attrFC.log & 
# # pid1=$!
# # python trainval.py --few_shot 0.3 --models bolt --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot03_hcpa_mlp1_attrFC.log & 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.3 --models bolt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot03_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.3 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot03_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.3 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 > logs3/boltFewShot03_abide_mlp1_attrFC.log & 
# pid5=$!
# python trainval.py --few_shot 0.3 --models bolt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot03_neurocon_mlp1_attrFC.log & 
# pid6=$!
# python trainval.py --few_shot 0.3 --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot03_taowu_mlp1_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7



# # python trainval.py --few_shot 0.5 --models bolt --dataname hcpya --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot05_hcpya_mlp1_attrFC.log & 
# # pid1=$!
# # python trainval.py --few_shot 0.5 --models bolt --dataname hcpa --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot05_hcpa_mlp1_attrFC.log & 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --few_shot 0.5 --models bolt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot05_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --few_shot 0.5 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot05_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --few_shot 0.5 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:4 --cv_fold_n 5 > logs3/boltFewShot05_abide_mlp1_attrFC.log & 
# pid5=$!
# python trainval.py --few_shot 0.5 --models bolt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot05_neurocon_mlp1_attrFC.log & 
# pid6=$!
# python trainval.py --few_shot 0.5 --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:5 --cv_fold_n 5 > logs3/boltFewShot05_taowu_mlp1_attrFC.log & 
# pid7=$!
# wait $pid5
# wait $pid6
# wait $pid7



# python trainval.py --few_shot 0.01 --models bolt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot001_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.01 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/boltFewShot001_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.01 --models bolt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/boltFewShot001_neurocon_mlp1_attrFC.log & 
python trainval.py --few_shot 0.01 --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/boltFewShot001_taowu_mlp1_attrFC.log 

# python trainval.py --few_shot 0.1 --models bolt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot01_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.1 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/boltFewShot01_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.1 --models bolt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/boltFewShot01_neurocon_mlp1_attrFC.log & 
python trainval.py --few_shot 0.1 --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/boltFewShot01_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.3 --models bolt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot03_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.3 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/boltFewShot03_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.3 --models bolt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/boltFewShot03_neurocon_mlp1_attrFC.log & 
python trainval.py --few_shot 0.3 --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/boltFewShot03_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.5 --models bolt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot05_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.5 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/boltFewShot05_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.5 --models bolt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/boltFewShot05_neurocon_mlp1_attrFC.log &
python trainval.py --few_shot 0.5 --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/boltFewShot05_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.01 --models bnt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot001_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.01 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot001_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.01 --models bnt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/bntFewShot001_neurocon_mlp1_attrFC.log & 
python trainval.py --few_shot 0.01 --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot001_taowu_mlp1_attrFC.log 

# python trainval.py --few_shot 0.1 --models bnt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot01_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.1 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot01_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.1 --models bnt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/bntFewShot01_neurocon_mlp1_attrFC.log & 
python trainval.py --few_shot 0.1 --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot01_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.3 --models bnt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot03_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.3 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot03_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.3 --models bnt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/bntFewShot03_neurocon_mlp1_attrFC.log & 
python trainval.py --few_shot 0.3 --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot03_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.5 --models bnt --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot05_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.5 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot05_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.5 --models bnt --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/bntFewShot05_neurocon_mlp1_attrFC.log &
python trainval.py --few_shot 0.5 --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/bntFewShot05_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.01 --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.01 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/neurodetourFewShot001_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.01 --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/neurodetourFewShot001_neurocon_mlp1_attrFC.log & 
python trainval.py --few_shot 0.01 --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/neurodetourFewShot001_taowu_mlp1_attrFC.log 

# python trainval.py --few_shot 0.1 --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.1 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/neurodetourFewShot01_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.1 --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/neurodetourFewShot01_neurocon_mlp1_attrFC.log & 
python trainval.py --few_shot 0.1 --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/neurodetourFewShot01_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.3 --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.3 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/neurodetourFewShot03_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.3 --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/neurodetourFewShot03_neurocon_mlp1_attrFC.log & 
python trainval.py --few_shot 0.3 --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/neurodetourFewShot03_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.5 --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.5 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/neurodetourFewShot05_abide_mlp1_attrFC.log 
python trainval.py --few_shot 0.5 --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/neurodetourFewShot05_neurocon_mlp1_attrFC.log &
python trainval.py --few_shot 0.5 --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/neurodetourFewShot05_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.01 --models braingnn --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot001_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.01 --models braingnn --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/braingnnFewShot001_abide_mlp1_attrFC.log 
# python trainval.py --few_shot 0.01 --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/braingnnFewShot001_neurocon_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.01 --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/braingnnFewShot001_taowu_mlp1_attrFC.log 

# python trainval.py --few_shot 0.1 --models braingnn --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot01_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.1 --models braingnn --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/braingnnFewShot01_abide_mlp1_attrFC.log 
# python trainval.py --few_shot 0.1 --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/braingnnFewShot01_neurocon_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.1 --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/braingnnFewShot01_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.3 --models braingnn --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot03_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.3 --models braingnn --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/braingnnFewShot03_abide_mlp1_attrFC.log 
# python trainval.py --few_shot 0.3 --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/braingnnFewShot03_neurocon_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.3 --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/braingnnFewShot03_taowu_mlp1_attrFC.log 


# python trainval.py --few_shot 0.5 --models braingnn --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot05_adni_mlp1_attrFC.log & 
# python trainval.py --few_shot 0.5 --models braingnn --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/braingnnFewShot05_abide_mlp1_attrFC.log 
# python trainval.py --few_shot 0.5 --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 > logs3/braingnnFewShot05_neurocon_mlp1_attrFC.log &
# python trainval.py --few_shot 0.5 --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 > logs3/braingnnFewShot05_taowu_mlp1_attrFC.log 



python trainval.py --few_shot 0.01 --models bolt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot001_sz-diana_mlp1_attrFC.log 

python trainval.py --few_shot 0.01 --models bnt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot001_sz-diana_mlp1_attrFC.log 

python trainval.py --few_shot 0.01 --models neurodetour --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_sz-diana_mlp1_attrFC.log 

python trainval.py --few_shot 0.01 --models graphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot001_sz-diana_mlp1_attrFC.log 

python trainval.py --few_shot 0.01 --models nagphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot001_sz-diana_mlp1_attrFC.log 


python trainval.py --few_shot 0.1 --models bolt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot01_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 0.1 --models bnt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot01_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 0.1 --models neurodetour --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 0.1 --models graphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot01_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 0.1 --models nagphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot01_sz-diana_mlp1_attrFC.log & 


python trainval.py --few_shot 0.3 --models bolt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot03_sz-diana_mlp1_attrFC.log & 


python trainval.py --few_shot 0.3 --models bnt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot03_sz-diana_mlp1_attrFC.log & 


python trainval.py --few_shot 0.3 --models neurodetour --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_sz-diana_mlp1_attrFC.log & 


python trainval.py --few_shot 0.3 --models graphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot03_sz-diana_mlp1_attrFC.log & 


python trainval.py --few_shot 0.3 --models nagphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot03_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 0.5 --models bolt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot05_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 0.5 --models bnt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot05_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 0.5 --models neurodetour --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 0.5 --models graphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot05_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 0.5 --models nagphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot05_sz-diana_mlp1_attrFC.log & 

python trainval.py --few_shot 1 --models bolt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot10_sz-diana_mlp1_attrFC.log & 


python trainval.py --few_shot 1 --models bnt --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot10_sz-diana_mlp1_attrFC.log & 


python trainval.py --few_shot 1 --models neurodetour --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot10_sz-diana_mlp1_attrFC.log & 


python trainval.py --few_shot 1 --models graphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot10_sz-diana_mlp1_attrFC.log & 


python trainval.py --few_shot 1 --models nagphormer --dataname sz-diana --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot10_sz-diana_mlp1_attrFC.log & 
