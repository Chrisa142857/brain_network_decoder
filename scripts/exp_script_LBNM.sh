nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:2 > logs/none_7datasets_decoder32_attrFC.log

nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:2 --savemodel --datanames hcpa hcpya > logs/none_hcpa-hcpya_decoder32_attrFC.log

nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:3 --savemodel --datanames hcpa hcpya adni > logs/none_hcpa-hcpya-adni_decoder32_attrFC.log

nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:5 --savemodel --datanames hcpa hcpya adni ppmi > logs/none_hcpa-hcpya-adni-ppmi_decoder32_attrFC.log

nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:0 --savemodel --datanames adni ppmi abide taowu neurocon > logs/none_adni-ppmi-abide-taowu-neurocon_decoder32_attrFC_v2.log


nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:4 --cv_fold_n 5 --savemodel --datanames adni ppmi abide taowu neurocon hcpa hcpya > logs/none_7datasets_decoder32Mix_attrFC.log

nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:2 --cv_fold_n 5 --savemodel --datanames adni ppmi abide taowu neurocon hcpa hcpya > logs/none_7datasets_decoder32_attrFC_v2.log

nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:3 --cv_fold_n 5 --savemodel --datanames adni ppmi abide taowu neurocon hcpa hcpya > logs/none_7datasets_decoder32_attrFC_v3.log

nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:4 --cv_fold_n 5 --savemodel --datanames adni ppmi abide taowu neurocon > logs/none_adni-ppmi-abide-taowu-neurocon_decoder32_attrFC_v3.log

nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 --savemodel --datanames hcpa hcpya > logs/none_hcpa-hcpya_decoder32_attrFC_v3.log

nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:2 --cv_fold_n 5 --savemodel --datanames ppmi abide taowu neurocon hcpa hcpya > logs/none_noAdni_decoder32_attrFC_v3.log
nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:3 --cv_fold_n 5 --savemodel --datanames adni abide taowu neurocon hcpa hcpya > logs/none_noPpmi_decoder32_attrFC_v3.log
nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:4 --cv_fold_n 5 --savemodel --datanames adni ppmi taowu neurocon hcpa hcpya > logs/none_noAbide_decoder32_attrFC_v3.log

nohup python trainval_lbnm.py --decoder --decoder_layer 8 --device cuda:3 --cv_fold_n 5 --savemodel --datanames ppmi abide taowu neurocon hcpa hcpya > logs/none_noAdni_decoder8_attrFC_v3.log
nohup python trainval_lbnm.py --decoder --decoder_layer 8 --device cuda:0 --cv_fold_n 5 --savemodel --datanames adni abide taowu neurocon hcpa hcpya > logs/none_noPpmi_decoder8_attrFC_v3.log
nohup python trainval_lbnm.py --decoder --decoder_layer 8 --device cuda:1 --cv_fold_n 5 --savemodel --datanames adni ppmi taowu neurocon hcpa hcpya > logs/none_noAbide_decoder8_attrFC_v3.log


nohup python trainval_lbnm.py --decoder --decoder_layer 32 --device cuda:5 --cv_fold_n 5 --datanames taowu adni ppmi abide neurocon hcpa hcpya --batch_size 32 --lr 0.00001 > logs/none_7datasets_decoder328Mix_attrFC_v3.log

nohup python trainval_lbnm.py --decoder --decoder_layer 8 --device cuda:1 --cv_fold_n 5 --datanames taowu adni ppmi abide neurocon hcpa hcpya --batch_size 32 --lr 0.00001 > logs/none_7datasets_decoder88Mix_attrFC_v3.log
