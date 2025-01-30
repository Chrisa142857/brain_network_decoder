# python trainval.py --models none --dataname ppmi --node_attr FC --device cuda:0 --decoder --cv_fold_n 10 > logs/none_ppmi_decoder88_attrFC.log & 
# python trainval.py --models none --dataname ppmi --node_attr BOLD --device cuda:1 --decoder --cv_fold_n 10 > logs/none_ppmi_decoder88_attrBOLD.log &
# python trainval.py --models gcn --dataname ppmi --node_attr FC --device cuda:2 --decoder --cv_fold_n 10 > logs/gcn_ppmi_decoder88_attrFC.log &
# python trainval.py --models gcn --dataname ppmi --node_attr BOLD --device cuda:3 --decoder --cv_fold_n 10 > logs/gcn_ppmi_decoder88_attrBOLD.log
# python trainval.py --models none --dataname abide --node_attr FC --device cuda:0 --decoder --cv_fold_n 10 > logs/none_abide_decoder88_attrFC.log & 
# python trainval.py --models none --dataname abide --node_attr BOLD --device cuda:1 --decoder --cv_fold_n 10 > logs/none_abide_decoder88_attrBOLD.log &
# nohup python trainval.py --models neurodetour --dataname ppmi --node_attr FC --device cuda:2 --decoder --cv_fold_n 10 --batch_size 512 > logs/neurodetour_ppmi_decoder88_attrFC.log & 
# nohup python trainval.py --models neurodetour --dataname abide --node_attr FC --device cuda:3 --decoder --cv_fold_n 10 --batch_size 512 > logs/neurodetour_abide_decoder88_attrFC.log & 
# nohup python trainval.py --models neurodetour --dataname ppmi --node_attr BOLD --device cuda:4 --decoder --cv_fold_n 10 --batch_size 512 > logs/neurodetour_ppmi_decoder88_attrBOLD.log & 
# nohup python trainval.py --models neurodetour --dataname abide --node_attr BOLD --device cuda:5 --decoder --cv_fold_n 10 --batch_size 512 > logs/neurodetour_abide_decoder88_attrBOLD.log 
# python trainval.py --models none --dataname ppmi --node_attr BOLD --device cuda:1 --decoder --cv_fold_n 10 > logs/none_ppmi_decoder88_attrBOLD.log &
# python trainval.py --models gcn --dataname ppmi --node_attr FC --device cuda:2 --decoder --cv_fold_n 10 > logs/gcn_ppmi_decoder88_attrFC.log &
# python trainval.py --models gcn --dataname ppmi --node_attr BOLD --device cuda:3 --decoder --cv_fold_n 10 > logs/gcn_ppmi_decoder88_attrBOLD.log

# python trainval.py --models none --dataname hcpa --node_attr FC --device cuda:0 --decoder --cv_fold_n 5 > logs/none_hcpa_decoder88_attrFC.log & 
# python trainval.py --models none --dataname hcpa --node_attr BOLD --device cuda:1 --decoder --cv_fold_n 5 > logs/none_hcpa_decoder88_attrBOLD.log &
# python trainval.py --models none --dataname adni --node_attr FC --device cuda:2 --decoder --cv_fold_n 5 > logs/none_adni_decoder88_attrFC.log &
# python trainval.py --models none --dataname adni --node_attr BOLD --device cuda:3 --decoder --cv_fold_n 5 > logs/none_adni_decoder88_attrBOLD.log 
# python trainval.py --models none --dataname oasis --node_attr FC --atlas D_160 --device cuda:4 --decoder --cv_fold_n 5 > logs/none_oasis_decoder88_attrFC.log &
# python trainval.py --models none --dataname oasis --node_attr BOLD --atlas D_160 --device cuda:5 --decoder --cv_fold_n 5 > logs/none_oasis_decoder88_attrBOLD.log


# python trainval.py --models none --dataname ppmi --node_attr FC --device cuda:0 --decoder --cv_fold_n 10 > logs/none_ppmi_mlp8_attrFC.log & 
# python trainval.py --models none --dataname ppmi --node_attr BOLD --device cuda:1 --decoder --cv_fold_n 10 > logs/none_ppmi_mlp8_attrBOLD.log &
# python trainval.py --models none --dataname abide --node_attr FC --device cuda:2 --decoder --cv_fold_n 10 > logs/none_abide_mlp8_attrFC.log & 
# python trainval.py --models none --dataname abide --node_attr BOLD --device cuda:3 --decoder --cv_fold_n 10 > logs/none_abide_mlp8_attrBOLD.log


# python trainval.py --models none --dataname hcpa --node_attr FC --device cuda:1 --decoder --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder328_attrFC.log & 
# python trainval.py --models none --dataname hcpa --node_attr BOLD --device cuda:3 --decoder --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_decoder328_attrBOLD.log 
# python trainval.py --models none --dataname adni --node_attr FC --device cuda:3 --decoder --cv_fold_n 5 --batch_size 256 > logs/none_adni_decoder328_attrFC.log &
# python trainval.py --models none --dataname adni --node_attr BOLD --device cuda:4 --decoder --cv_fold_n 5 --batch_size 256 > logs/none_adni_decoder328_attrBOLD.log 
# python trainval.py --models none --dataname oasis --node_attr FC --atlas D_160 --device cuda:4 --decoder --cv_fold_n 5 --batch_size 128 > logs/none_oasis_decoder328_attrFC.log &
# python trainval.py --models none --dataname oasis --node_attr BOLD --atlas D_160 --device cuda:5 --decoder --cv_fold_n 5 --batch_size 128 > logs/none_oasis_decoder328_attrBOLD.log

# nohup python trainval.py --models none --dataname ppmi --node_attr FC --device cuda:1 --decoder --cv_fold_n 10 > logs/none_ppmi_decoder328_attrFC.log & 
# nohup python trainval.py --models none --dataname ppmi --node_attr BOLD --device cuda:3 --decoder --cv_fold_n 10 > logs/none_ppmi_decoder328_attrBOLD.log 
# nohup python trainval.py --models none --dataname abide --node_attr FC --device cuda:4 --decoder --cv_fold_n 10 --batch_size 256 > logs/none_abide_decoder328_attrFC.log & 
# nohup python trainval.py --models none --dataname abide --node_attr BOLD --device cuda:5 --decoder --cv_fold_n 10 --batch_size 256 > logs/none_abide_decoder328_attrBOLD.log


# python trainval.py --models none --dataname hcpya --node_attr FC --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp8_attrFC.log & 
# python trainval.py --models none --dataname hcpya --node_attr BOLD --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp8_attrBOLD.log & 
# python trainval.py --models none --dataname hcpya --node_attr FC --device cuda:4 --decoder --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder328_attrFC.log & 
# python trainval.py --models none --dataname hcpya --node_attr BOLD --device cuda:5 --decoder --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_decoder328_attrBOLD.log 
# python trainval.py --models none --dataname hcpya --node_attr FC --device cuda:0 --decoder --cv_fold_n 5 --batch_size 128 --decoder_layer 8 > logs/none_hcpya_decoder88_attrFC.log & 
# python trainval.py --models none --dataname hcpya --node_attr BOLD --device cuda:1 --decoder --cv_fold_n 5 --batch_size 128 --decoder_layer 8 > logs/none_hcpya_decoder88_attrBOLD.log 


# python trainval.py --models none --dataname hcpya --node_attr FC --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp8Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpya --node_attr FC --device cuda:2 --decoder --cv_fold_n 5 --batch_size 128 --decoder_layer 8 > logs/none_hcpya_decoder88Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp8Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --device cuda:4 --decoder --cv_fold_n 5 --batch_size 128 --decoder_layer 8 > logs/none_hcpa_decoder88Mix_attrFC.log &
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname adni --node_attr FC --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp8Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname adni --node_attr FC --device cuda:2 --decoder --cv_fold_n 5 --batch_size 128 --decoder_layer 8 > logs/none_adni_decoder88Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --device cuda:3 --decoder --cv_fold_n 5 > logs/none_ppmi_decoder88Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --device cuda:4 --cv_fold_n 5 > logs/none_ppmi_mlp8Mix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --device cuda:1 --decoder --cv_fold_n 5 > logs/none_abide_decoder88Mix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname abide --node_attr FC --device cuda:2 --cv_fold_n 5 > logs/none_abide_mlp8Mix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --device cuda:3 --decoder --cv_fold_n 5 > logs/none_neurocon_decoder88Mix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname neurocon --node_attr FC --device cuda:4 --cv_fold_n 5 > logs/none_neurocon_mlp8Mix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname taowu --node_attr FC --device cuda:1 --decoder --cv_fold_n 5 > logs/none_taowu_decoder88Mix_attrFC.log & 
# python trainval.py --models none --dataname taowu --node_attr FC --device cuda:2 --cv_fold_n 5 > logs/none_taowu_mlp8Mix_attrFC.log 



# python trainval.py --models none --dataname hcpya --node_attr FC --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp8L_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp8L_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp8L_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --device cuda:4 --cv_fold_n 5 > logs/none_ppmi_mlp8L_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --device cuda:1 --cv_fold_n 5 > logs/none_abide_mlp8L_attrFC.log & 
# python trainval.py --models none --dataname neurocon --node_attr FC --device cuda:2 --cv_fold_n 5 > logs/none_neurocon_mlp8L_attrFC.log & 
# python trainval.py --models none --dataname taowu --node_attr FC --device cuda:3 --cv_fold_n 5 > logs/none_taowu_mlp8L_attrFC.log 



# python trainval.py --models none --dataname hcpya --node_attr FC --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs/none_hcpya_mlp8LMix_attrFC.log & 
# pid1=$!
# python trainval.py --models none --dataname hcpa --node_attr FC --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs/none_hcpa_mlp8LMix_attrFC.log & 
# pid2=$!
# python trainval.py --models none --dataname adni --node_attr FC --device cuda:3 --cv_fold_n 5 --batch_size 128 > logs/none_adni_mlp8LMix_attrFC.log & 
# pid3=$!
# python trainval.py --models none --dataname ppmi --node_attr FC --device cuda:4 --cv_fold_n 10 > logs/none_ppmi_mlp8LMix_attrFC.log & 
# pid4=$!
# wait $pid1
# wait $pid2
# wait $pid3
# wait $pid4
# python trainval.py --models none --dataname abide --node_attr FC --device cuda:1 --cv_fold_n 10 > logs/none_abide_mlp8LMix_attrFC.log & 
# python trainval.py --models none --dataname neurocon --node_attr FC --device cuda:2 --cv_fold_n 10 > logs/none_neurocon_mlp8LMix_attrFC.log & 
# python trainval.py --models none --dataname taowu --node_attr FC --device cuda:3 --cv_fold_n 10 > logs/none_taowu_mlp8LMix_attrFC.log 
