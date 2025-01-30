
# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 4 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp4LMix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 4 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp4LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 4 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp4LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 4 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp4LMix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 4 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp4LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 4 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp4LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 4 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp4LMix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 12 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp12LMix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 12 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp12LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 12 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp12LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 12 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp12LMix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 12 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp12LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 12 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp12LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 12 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp12LMix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 16 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp16LMix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 16 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp16LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 16 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp16LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 16 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp16LMix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 16 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp16LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 16 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp16LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 16 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp16LMix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 20 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp20LMix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 20 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp20LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 20 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp20LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 20 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp20LMix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 20 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp20LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 20 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp20LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 20 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp20LMix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 24 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp24LMix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 24 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp24LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp24LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp24LMix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 24 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp24LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 24 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp24LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp24LMix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 28 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp28LMix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 28 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp28LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 28 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp28LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 28 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp28LMix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 28 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp28LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 28 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp28LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 28 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp28LMix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp32LMix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 32 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp32LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 32 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp32LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 32 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp32LMix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 32 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp32LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 32 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp32LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 32 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp32LMix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 4 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp4_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 4 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp4_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 4 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp4_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 4 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp4_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 4 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp4_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 4 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp4_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 4 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp4_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 12 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp12_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 12 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp12_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 12 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp12_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 12 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp12_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 12 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp12_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 12 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp12_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 12 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp12_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 16 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp16_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 16 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp16_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 16 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp16_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 16 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp16_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 16 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp16_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 16 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp16_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 16 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp16_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 20 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp20_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 20 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp20_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 20 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp20_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 20 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp20_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 20 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp20_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 20 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp20_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 20 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp20_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 24 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp24_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 24 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp24_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp24_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp24_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 24 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp24_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 24 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp24_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp24_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 28 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp28_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 28 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp28_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 28 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp28_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 28 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp28_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 28 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp28_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 28 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp28_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 28 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp28_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 32 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp32_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 32 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp32_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 32 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp32_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 32 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp32_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 32 --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp32_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 32 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp32_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 32 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp32_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 4 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpya_MLP4_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 4 --hiddim 6144 --device cuda:1 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpa_MLP4_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 4 --hiddim 6144 --device cuda:2 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_adni_MLP4_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 4 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_ppmi_MLP4_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 4 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_MLP4_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 4 --hiddim 6144 --device cuda:2 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_neurocon_MLP4_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 4 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_taowu_MLP4_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 8 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpya_MLP8_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 8 --hiddim 6144 --device cuda:1 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpa_MLP8_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 8 --hiddim 6144 --device cuda:2 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_adni_MLP8_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 8 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_ppmi_MLP8_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 8 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_MLP8_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 8 --hiddim 6144 --device cuda:2 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_neurocon_MLP8_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 8 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_taowu_MLP8_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 12 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpya_MLP12_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 12 --hiddim 6144 --device cuda:1 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpa_MLP12_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 12 --hiddim 6144 --device cuda:2 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_adni_MLP12_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 12 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_ppmi_MLP12_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 12 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_MLP12_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 12 --hiddim 6144 --device cuda:2 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_neurocon_MLP12_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 12 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_taowu_MLP12_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 16 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpya_MLP16_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 16 --hiddim 6144 --device cuda:1 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpa_MLP16_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 16 --hiddim 6144 --device cuda:2 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_adni_MLP16_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 16 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_ppmi_MLP16_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 16 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_MLP16_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 16 --hiddim 6144 --device cuda:2 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_neurocon_MLP16_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 16 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_taowu_MLP16_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 20 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpya_MLP20_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 20 --hiddim 6144 --device cuda:1 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpa_MLP20_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 20 --hiddim 6144 --device cuda:2 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_adni_MLP20_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 20 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_ppmi_MLP20_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 20 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_MLP20_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 20 --hiddim 6144 --device cuda:2 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_neurocon_MLP20_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 20 --hiddim 6144 --device cuda:3 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_taowu_MLP20_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3

