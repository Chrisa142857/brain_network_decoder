
# python trainval.py --models graphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs/graphormer_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 10 > logs/graphormer_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4
# python trainval.py --models graphormer --dataname abide --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 10 > logs/graphormer_abide_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --models nagphormer --dataname adni --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/nagphormer_adni_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4


# python trainval.py --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 10 > logs/nagphormer_ppmi_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --models nagphormer --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 10 > logs/nagphormer_taowu_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4


# python trainval.py --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_adni_mlp1_attrFC.log & 
# pid3=$!
# python trainval.py --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 10 > logs/neurodetour_ppmi_mlp1_attrFC.log & 
# pid4=$!
# wait $pid3
# wait $pid4

# python trainval.py --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 10 > logs/neurodetour_taowu_mlp1_attrFC.log & 
# pid1=$!
# wait $pid1



# python trainval.py --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/bnt_taowu_mlp1Mix_attrFC.log 
# # pid1=$!
# python trainval.py --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 --batch_size 32 --lr 0.000001 > logs/braingnn_taowu_mlp1Mix_attrFC.log 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/bolt_taowu_mlp1Mix_attrFC.log 
# # pid1=$!
# python trainval.py --models graphormer --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/graphormer_taowu_mlp1Mix_attrFC.log 
# # pid2=$!
# # wait $pid1
# # wait $pid2
# python trainval.py --models nagphormer --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/nagphormer_taowu_mlp1Mix_attrFC.log 
# # pid1=$!
# python trainval.py --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/neurodetour_taowu_mlp1Mix_attrFC.log 
# # pid2=$!
# # wait $pid1
# # wait $pid2


# python trainval.py --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 --batch_size 16 --lr 0.000001 > logs/braingnn_neurocon_mlp1Mix_attrFC.log

# python trainval.py --models none --dataname taowu --node_attr FC --decoder --decoder_layer 8 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/none_taowuSexAge_decoder88Mix_attrFC.log 
python trainval.py --models neurodetour --dataname hcpya --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_hcpya_mlp1Mix_attrFC.log 
python trainval.py --models neurodetour --dataname hcpa --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_hcpa_mlp1Mix_attrFC.log 
python trainval.py --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_adni_mlp1Mix_attrFC.log 
python trainval.py --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/neurodetour_ppmi_mlp1Mix_attrFC.log 
python trainval.py --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/neurodetour_abide_mlp1Mix_attrFC.log 
python trainval.py --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/neurodetour_neurocon_mlp1Mix_attrFC.log 
