
# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 28 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder288Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 28 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder288Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 28 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder288Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 28 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder288Mix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 28 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder288Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 28 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder288Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 28 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder288Mix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder248Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder248Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder248Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder248Mix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder248Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder248Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder248Mix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3

# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 20 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder208Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 20 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder208Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 20 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder208Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 20 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder208Mix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 20 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder208Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 20 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder208Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 20 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder208Mix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 16 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder168Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 16 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder168Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 16 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder168Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 16 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder168Mix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
#python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 16 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder168Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 16 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder168Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 16 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder168Mix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 12 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder128Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 12 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder128Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 12 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder128Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 12 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder128Mix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 12 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder128Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 12 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder128Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 12 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder128Mix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 4 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder48Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 4 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder48Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 4 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder48Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 4 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder48Mix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 4 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder48Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 4 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder48Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 4 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder48Mix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 28 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder28_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 28 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder28_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 28 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder28_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 28 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder28_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 28 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder28_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 28 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder28_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 28 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder28_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder24_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder24_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder24_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder24_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder24_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3

# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 20 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder20_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 20 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder20_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 20 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder20_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 20 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder20_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 20 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder20_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 20 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder20_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 20 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder20_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 16 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder16_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 16 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder16_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 16 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder16_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 16 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder16_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 16 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder16_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 16 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder16_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 16 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder16_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 12 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder12_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 12 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder12_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 12 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder12_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 12 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder12_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 12 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder12_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 12 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder12_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 12 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder12_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 4 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder4_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 4 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder4_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 4 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder4_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 4 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder4_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 4 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder4_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 4 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder4_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 4 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder4_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3

# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 8 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder8_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 8 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder8_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 8 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_decoder8_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 8 --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_decoder8_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --batch_size 512 --node_attr FC --decoder --decoder_layer 8 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder8_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 8 --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_decoder8_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 8 --device cuda:3 --cv_fold_n 10 > logs/none_taowu_decoder8_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname abide --batch_size 256 --node_attr FC --decoder --decoder_layer 28 --device cuda:4 --cv_fold_n 10 > logs/none_abide_decoder28_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4

# python trainval.py --models none --dataname abide --batch_size 256 --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 10 > logs/none_abide_decoder24_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname abide --batch_size 256 --node_attr FC --decoder --decoder_layer 20 --device cuda:2 --cv_fold_n 10 > logs/none_abide_decoder20_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname abide --batch_size 512 --node_attr FC --decoder --decoder_layer 16 --device cuda:3 --cv_fold_n 10 > logs/none_abide_decoder16_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname abide --batch_size 512 --node_attr FC --decoder --decoder_layer 12 --device cuda:4 --cv_fold_n 10 > logs/none_abide_decoder12_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4


python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 28 --device cuda:0 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_decoder288Mix_attrFC.log & 
pid1=$!
python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_decoder248Mix_attrFC.log & 
pid2=$!
wait $pid1
wait $pid2
python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 20 --device cuda:0 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_decoder208Mix_attrFC.log & 
pid1=$!
python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 32 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_decoder328Mix_attrFC.log &
pid2=$!
wait $pid1
wait $pid2

