

# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 24 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpya_MLP24_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 24 --hiddim 6144 --device cuda:1 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpa_MLP24_attrFC.log & 
# pid2=$!
# wait $pid1
# wait $pid2
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 24 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_adni_MLP24_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 24 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_ppmi_MLP24_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 24 --hiddim 6144 --device cuda:0 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_MLP24_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 24 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_neurocon_MLP24_attrFC.log & 
# pid2=$!
# wait $pid2
# wait $pid3


# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 24 --hiddim 6144 --device cuda:0 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_taowu_MLP24_attrFC.log & 
# pid0=$!
# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 28 --hiddim 6144 --device cuda:1 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpya_MLP28_attrFC.log & 
# pid1=$!
# wait $pid0
# wait $pid1
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 28 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpa_MLP28_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 28 --hiddim 6144 --device cuda:1 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_adni_MLP28_attrFC.log & 
# pid3=$!
# wait $pid2
# wait $pid3
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 28 --hiddim 6144 --device cuda:0 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_ppmi_MLP28_attrFC.log & 
# pid4=$!
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 28 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_MLP28_attrFC.log & 
# pid3=$!
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 28 --hiddim 6144 --device cuda:0 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_neurocon_MLP28_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 28 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_taowu_MLP28_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2


# python trainval.py --models none --dataname hcpya --node_attr FC --decoder_layer 32 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpya_MLP32_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder_layer 32 --hiddim 6144 --device cuda:1 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_hcpa_MLP32_attrFC.log & 
# pid2=$!
# wait $pid1
# wait $pid2
# python trainval.py --models none --dataname adni --node_attr FC --decoder_layer 32 --hiddim 6144 --device cuda:0 --cv_fold_n 5 --batch_size 32 --lr 0.00001 > logs/none_adni_MLP32_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder_layer 32 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_ppmi_MLP32_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --decoder_layer 32 --hiddim 6144 --device cuda:0 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_MLP32_attrFC.log & 
# pid3=$!
# # python trainval.py --models none --dataname neurocon --node_attr FC --decoder_layer 32 --hiddim 6144 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_neurocon_MLP32_attrFC.log & 
# # pid2=$!
# # wait $pid2
# wait $pid3
# # python trainval.py --models none --dataname taowu --node_attr FC --decoder_layer 32 --hiddim 6144 --device cuda:0 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_taowu_MLP32_attrFC.log & 
# # pid1=$!
# # wait $pid1


python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 4 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_decoder48Mix_attrFC.log
python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 8 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_decoder88Mix_attrFC.log 
python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 12 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_decoder128Mix_attrFC.log
python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 16 --device cuda:1 --cv_fold_n 10 --batch_size 128 --lr 0.00001 > logs/none_abide_decoder168Mix_attrFC.log 



