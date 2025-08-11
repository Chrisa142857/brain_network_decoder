python trainval.py --models none --dataname neurocon --node_attr FC --device cuda:4 --decoder --cv_fold_n 5 > logs/none_neurocon_decoder328_attrFC.log & 
python trainval.py --models none --dataname neurocon --node_attr BOLD --device cuda:5 --decoder --cv_fold_n 5 > logs/none_neurocon_decoder328_attrBOLD.log
python trainval.py --models gcn --dataname neurocon --node_attr FC --device cuda:4 --decoder --cv_fold_n 5 > logs/gcn_neurocon_decoder328_attrFC.log &
python trainval.py --models gcn --dataname neurocon --node_attr BOLD --device cuda:5 --decoder --cv_fold_n 5 > logs/gcn_neurocon_decoder328_attrBOLD.log
python trainval.py --models neurodetour --dataname neurocon --node_attr FC --device cuda:4 --decoder --cv_fold_n 5 > logs/neurodetour_neurocon_decoder328_attrFC.log &
python trainval.py --models neurodetour --dataname neurocon --node_attr BOLD --device cuda:5 --decoder --cv_fold_n 5 > logs/neurodetour_neurocon_decoder328_attrBOLD.log

python trainval.py --models none --dataname neurocon --node_attr FC --device cuda:4 --cv_fold_n 5 > logs/none_neurocon_mlp8_attrFC.log & 
python trainval.py --models none --dataname neurocon --node_attr BOLD --device cuda:5 --cv_fold_n 5 > logs/none_neurocon_mlp8_attrBOLD.log
python trainval.py --models gcn --dataname neurocon --node_attr FC --device cuda:4 --cv_fold_n 5 > logs/gcn_neurocon_mlp8_attrFC.log &
python trainval.py --models gcn --dataname neurocon --node_attr BOLD --device cuda:5 --cv_fold_n 5 > logs/gcn_neurocon_mlp8_attrBOLD.log
python trainval.py --models neurodetour --dataname neurocon --node_attr FC --device cuda:4 --cv_fold_n 5 > logs/neurodetour_neurocon_mlp8_attrFC.log &
python trainval.py --models neurodetour --dataname neurocon --node_attr BOLD --device cuda:5 --cv_fold_n 5 > logs/neurodetour_neurocon_mlp8_attrBOLD.log

python trainval.py --models none --dataname taowu --node_attr FC --device cuda:4 --decoder --cv_fold_n 5 > logs/none_taowu_decoder328_attrFC.log & 
python trainval.py --models none --dataname taowu --node_attr BOLD --device cuda:5 --decoder --cv_fold_n 5 > logs/none_taowu_decoder328_attrBOLD.log
python trainval.py --models gcn --dataname taowu --node_attr FC --device cuda:4 --decoder --cv_fold_n 5 > logs/gcn_taowu_decoder328_attrFC.log &
python trainval.py --models gcn --dataname taowu --node_attr BOLD --device cuda:5 --decoder --cv_fold_n 5 > logs/gcn_taowu_decoder328_attrBOLD.log
python trainval.py --models neurodetour --dataname taowu --node_attr FC --device cuda:4 --decoder --cv_fold_n 5 > logs/neurodetour_taowu_decoder328_attrFC.log &
python trainval.py --models neurodetour --dataname taowu --node_attr BOLD --device cuda:5 --decoder --cv_fold_n 5 > logs/neurodetour_taowu_decoder328_attrBOLD.log

python trainval.py --models none --dataname taowu --node_attr FC --device cuda:4 --cv_fold_n 5 > logs/none_taowu_mlp8_attrFC.log & 
python trainval.py --models none --dataname taowu --node_attr BOLD --device cuda:5 --cv_fold_n 5 > logs/none_taowu_mlp8_attrBOLD.log
python trainval.py --models gcn --dataname taowu --node_attr FC --device cuda:4 --cv_fold_n 5 > logs/gcn_taowu_mlp8_attrFC.log &
python trainval.py --models gcn --dataname taowu --node_attr BOLD --device cuda:5 --cv_fold_n 5 > logs/gcn_taowu_mlp8_attrBOLD.log
python trainval.py --models neurodetour --dataname taowu --node_attr FC --device cuda:4 --cv_fold_n 5 > logs/neurodetour_taowu_mlp8_attrFC.log &
python trainval.py --models neurodetour --dataname taowu --node_attr BOLD --device cuda:5 --cv_fold_n 5 > logs/neurodetour_taowu_mlp8_attrBOLD.log
