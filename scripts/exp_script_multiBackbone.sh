
python trainval.py --models bnt --dataname hcpya --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs/bnt_hcpya_mlp1Mix_attrFC.log & 
pid1=$!
python trainval.py --models bnt --dataname hcpa --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/bnt_hcpa_mlp1Mix_attrFC.log & 
pid2=$!
python trainval.py --models bnt --dataname adni --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/bnt_adni_mlp1Mix_attrFC.log & 
pid3=$!
python trainval.py --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/bnt_ppmi_mlp1Mix_attrFC.log & 
pid4=$!
python trainval.py --models bnt --dataname abide --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:5 --cv_fold_n 10 > logs/bnt_abide_mlp1Mix_attrFC.log & 
pid5=$!
python trainval.py --models bnt --dataname neurocon --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:4 --cv_fold_n 10 > logs/bnt_neurocon_mlp1Mix_attrFC.log & 
pid6=$!
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6
# python trainval.py --models bnt --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/bnt_taowu_mlp1Mix_attrFC.log & 
# pid1=$!
# wait $pid1



python trainval.py --models braingnn --dataname hcpya --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:0 --cv_fold_n 5 --batch_size 2 --lr 0.000001 > logs/braingnn_hcpya_mlp1Mix_attrFC.log & 
pid1=$!
python trainval.py --models braingnn --dataname hcpa --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:2 --cv_fold_n 5 --batch_size 2 --lr 0.000001 > logs/braingnn_hcpa_mlp1Mix_attrFC.log & 
pid2=$!
python trainval.py --models braingnn --dataname adni --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 2 --lr 0.000001 > logs/braingnn_adni_mlp1Mix_attrFC.log & 
pid3=$!
python trainval.py --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 --batch_size 2 --lr 0.000001 > logs/braingnn_ppmi_mlp1Mix_attrFC.log & 
pid4=$!
python trainval.py --models braingnn --dataname abide --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:5 --cv_fold_n 10 --batch_size 2 --lr 0.000001 > logs/braingnn_abide_mlp1Mix_attrFC.log & 
pid5=$!
python trainval.py --models braingnn --dataname neurocon --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:4 --cv_fold_n 10 > logs/braingnn_neurocon_mlp1Mix_attrFC.log & 
pid6=$!
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6
# python trainval.py --models braingnn --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:0 --cv_fold_n 10 > logs/braingnn_taowu_mlp1Mix_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models bolt --dataname hcpya --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs/bolt_hcpya_mlp1Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models bolt --dataname hcpa --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/bolt_hcpa_mlp1Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models bolt --dataname adni --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/bolt_adni_mlp1Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/bolt_ppmi_mlp1Mix_attrFC.log & 
# pid4=$!
# python trainval.py --models bolt --dataname abide --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:5 --cv_fold_n 10 > logs/bolt_abide_mlp1Mix_attrFC.log & 
# pid5=$!
# python trainval.py --models bolt --dataname neurocon --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:4 --cv_fold_n 10 > logs/bolt_neurocon_mlp1Mix_attrFC.log & 
# pid6=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# wait $pid5
# wait $pid6
# python trainval.py --models bolt --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/bolt_taowu_mlp1Mix_attrFC.log & 
# pid1=$!
# wait $pid1



# python trainval.py --models graphormer --dataname hcpya --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/graphormer_hcpya_mlp1Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models graphormer --dataname hcpa --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/graphormer_hcpa_mlp1Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models graphormer --dataname adni --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/graphormer_adni_mlp1Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/graphormer_ppmi_mlp1Mix_attrFC.log & 
# pid4=$!
# python trainval.py --models graphormer --dataname abide --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:4 --cv_fold_n 10 > logs/graphormer_abide_mlp1Mix_attrFC.log & 
# pid5=$!
# python trainval.py --models graphormer --dataname neurocon --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/graphormer_neurocon_mlp1Mix_attrFC.log & 
# pid6=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# wait $pid5
# wait $pid6
# python trainval.py --models graphormer --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/graphormer_taowu_mlp1Mix_attrFC.log & 
# pid1=$!
# wait $pid1



# python trainval.py --models nagphormer --dataname hcpya --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/nagphormer_hcpya_mlp1Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models nagphormer --dataname hcpa --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs/nagphormer_hcpa_mlp1Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models nagphormer --dataname adni --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/nagphormer_adni_mlp1Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:4 --cv_fold_n 10 > logs/nagphormer_ppmi_mlp1Mix_attrFC.log & 
# pid4=$!
# python trainval.py --models nagphormer --dataname abide --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/nagphormer_abide_mlp1Mix_attrFC.log & 
# pid5=$!
# python trainval.py --models nagphormer --dataname neurocon --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/nagphormer_neurocon_mlp1Mix_attrFC.log & 
# pid6=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# wait $pid5
# wait $pid6
# python trainval.py --models nagphormer --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/nagphormer_taowu_mlp1Mix_attrFC.log & 
# pid1=$!
# wait $pid1


# python trainval.py --models neurodetour --dataname hcpya --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_hcpya_mlp1Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models neurodetour --dataname hcpa --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_hcpa_mlp1Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models neurodetour --dataname adni --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_adni_mlp1Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/neurodetour_ppmi_mlp1Mix_attrFC.log & 
# pid4=$!
# python trainval.py --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:5 --cv_fold_n 10 > logs/neurodetour_abide_mlp1Mix_attrFC.log & 
# pid5=$!
# python trainval.py --models neurodetour --dataname neurocon --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:4 --cv_fold_n 10 > logs/neurodetour_neurocon_mlp1Mix_attrFC.log & 
# pid6=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# wait $pid5
# wait $pid6
# python trainval.py --models neurodetour --dataname taowu --node_attr FC --decoder_layer 1 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/neurodetour_taowu_mlp1Mix_attrFC.log & 
# pid1=$!
# wait $pid1

# python trainval.py --models none --dataname hcpya --node_attr FC --decoder --decoder_layer 8 --train_obj sex --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs/none_hcpyaSexAge_decoder88Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --decoder --decoder_layer 8 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpaSexAge_decoder88Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --decoder --decoder_layer 8 --train_obj sex --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_adniSexAge_decoder88Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --decoder --decoder_layer 8 --train_obj sex --device cuda:1 --cv_fold_n 10 > logs/none_ppmiSexAge_decoder88Mix_attrFC.log & 
# pid4=$!
# python trainval.py --models none --dataname abide --node_attr FC --decoder --decoder_layer 8 --train_obj sex --device cuda:3 --cv_fold_n 10 > logs/none_abideSexAge_decoder88Mix_attrFC.log & 
# pid5=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --decoder --decoder_layer 8 --train_obj sex --device cuda:4 --cv_fold_n 10 > logs/none_neuroconSexAge_decoder88Mix_attrFC.log & 
# pid6=$!
wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6

## TODO: Debug below exp
# python trainval.py --models bnt --dataname hcpya --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/bnt_hcpya_decoder24_attrFC.log & 
# pid1=$!
# python trainval.py --models bnt --dataname hcpa --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/bnt_hcpa_decoder24_attrFC.log & 
# pid2=$!
# python trainval.py --models bnt --dataname adni --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/bnt_adni_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models bnt --dataname ppmi --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/bnt_ppmi_decoder24_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models bnt --dataname abide --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 10 > logs/bnt_abide_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models bnt --dataname neurocon --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 10 > logs/bnt_neurocon_decoder24_attrFC.log & 
# pid2=$!
# python trainval.py --models bnt --dataname taowu --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/bnt_taowu_decoder24_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models braingnn --dataname hcpya --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/braingnn_hcpya_decoder24_attrFC.log & 
# pid1=$!
# python trainval.py --models braingnn --dataname hcpa --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/braingnn_hcpa_decoder24_attrFC.log & 
# pid2=$!
# python trainval.py --models braingnn --dataname adni --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/braingnn_adni_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models braingnn --dataname ppmi --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/braingnn_ppmi_decoder24_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models braingnn --dataname abide --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 10 > logs/braingnn_abide_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models braingnn --dataname neurocon --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 10 > logs/braingnn_neurocon_decoder24_attrFC.log & 
# pid2=$!
# python trainval.py --models braingnn --dataname taowu --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/braingnn_taowu_decoder24_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models bolt --dataname hcpya --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/bolt_hcpya_decoder24_attrFC.log & 
# pid1=$!
# python trainval.py --models bolt --dataname hcpa --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/bolt_hcpa_decoder24_attrFC.log & 
# pid2=$!
# python trainval.py --models bolt --dataname adni --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/bolt_adni_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models bolt --dataname ppmi --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/bolt_ppmi_decoder24_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models bolt --dataname abide --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 10 > logs/bolt_abide_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models bolt --dataname neurocon --node_attr FC --decoder --decoder_layer 24 --device cuda:2 --cv_fold_n 10 > logs/bolt_neurocon_decoder24_attrFC.log & 
# pid2=$!
# python trainval.py --models bolt --dataname taowu --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/bolt_taowu_decoder24_attrFC.log & 
# pid1=$!
# wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models graphormer --dataname hcpya --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/graphormer_hcpya_decoder24_attrFC.log & 
# pid1=$!
# python trainval.py --models graphormer --dataname hcpa --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs/graphormer_hcpa_decoder24_attrFC.log & 
# pid2=$!
# # python trainval.py --models graphormer --dataname adni --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/graphormer_adni_decoder24_attrFC.log & 
# # pid3=$!
# # python trainval.py --models graphormer --dataname ppmi --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/graphormer_ppmi_decoder24_attrFC.log & 
# # pid4=$!
# wait $pid1
# wait $pid2
# # wait $pid3
# # wait $pid4
# # python trainval.py --models graphormer --dataname abide --node_attr FC --decoder --decoder_layer 24 --device cuda:1 --cv_fold_n 10 > logs/graphormer_abide_decoder24_attrFC.log & 
# # pid3=$!
# python trainval.py --models graphormer --dataname neurocon --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/graphormer_neurocon_decoder24_attrFC.log & 
# pid2=$!
# python trainval.py --models graphormer --dataname taowu --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/graphormer_taowu_decoder24_attrFC.log & 
# pid1=$!
# # wait $pid1
# wait $pid2
# wait $pid3



# python trainval.py --models nagphormer --dataname hcpya --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/nagphormer_hcpya_decoder24_attrFC.log & 
# pid1=$!
# python trainval.py --models nagphormer --dataname hcpa --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs/nagphormer_hcpa_decoder24_attrFC.log & 
# pid2=$!
# # python trainval.py --models nagphormer --dataname adni --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/nagphormer_adni_decoder24_attrFC.log & 
# # pid3=$!
# # python trainval.py --models nagphormer --dataname ppmi --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/nagphormer_ppmi_decoder24_attrFC.log & 
# # pid4=$!
# wait $pid1
# wait $pid2
# # wait $pid3
# # wait $pid4
# python trainval.py --models nagphormer --dataname abide --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/nagphormer_abide_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models nagphormer --dataname neurocon --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/nagphormer_neurocon_decoder24_attrFC.log & 
# pid2=$!
# # python trainval.py --models nagphormer --dataname taowu --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/nagphormer_taowu_decoder24_attrFC.log & 
# # pid1=$!
# # wait $pid1
# wait $pid2
# wait $pid3


# python trainval.py --models neurodetour --dataname hcpya --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_hcpya_decoder24_attrFC.log & 
# pid1=$!
# python trainval.py --models neurodetour --dataname hcpa --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_hcpa_decoder24_attrFC.log & 
# pid2=$!
# # python trainval.py --models neurodetour --dataname adni --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/neurodetour_adni_decoder24_attrFC.log & 
# # pid3=$!
# # python trainval.py --models neurodetour --dataname ppmi --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/neurodetour_ppmi_decoder24_attrFC.log & 
# # pid4=$!
# wait $pid1
# wait $pid2
# # wait $pid3
# # wait $pid4
# python trainval.py --models neurodetour --dataname abide --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/neurodetour_abide_decoder24_attrFC.log & 
# pid3=$!
# python trainval.py --models neurodetour --dataname neurocon --node_attr FC --decoder --decoder_layer 24 --device cuda:4 --cv_fold_n 10 > logs/neurodetour_neurocon_decoder24_attrFC.log & 
# pid2=$!
# # python trainval.py --models neurodetour --dataname taowu --node_attr FC --decoder --decoder_layer 24 --device cuda:3 --cv_fold_n 10 > logs/neurodetour_taowu_decoder24_attrFC.log & 
# # pid1=$!
# # wait $pid1
# wait $pid2
# wait $pid3


