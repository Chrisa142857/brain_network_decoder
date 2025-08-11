python trainval.py --dataname adni --decoder --device cuda:0 --epoch 60 > logs_moe/none_adni_decoder328_attrBOLDwin160.log
python trainval.py --dataname ppmi --decoder --device cuda:0 --epoch 60 > logs_moe/none_ppmi_decoder328_attrBOLDwin160.log
python trainval.py --dataname taowu --decoder --device cuda:0 --epoch 60 > logs_moe/none_taowu_decoder328_attrBOLDwin160.log
python trainval.py --dataname neurocon --decoder --device cuda:0 --epoch 60 > logs_moe/none_neurocon_decoder328_attrBOLDwin160.log
python trainval.py --dataname sz-diana --decoder --device cuda:0 --epoch 60 > logs_moe/none_sz-diana_decoder328_attrBOLDwin160.log
python trainval.py --dataname abide --decoder --device cuda:1 --epoch 60 > logs_moe/none_abide_decoder328_attrBOLDwin160.log

python trainval.py --dataname adni --train_obj sex --decoder --device cuda:0 --epoch 60 > logs_moe/none_adni_decoder328_attrBOLDwin160.log
python trainval.py --dataname ppmi --train_obj sex --decoder --device cuda:0 --epoch 60 > logs_moe/none_ppmi_decoder328_attrBOLDwin160.log
python trainval.py --dataname taowu --train_obj sex --decoder --device cuda:0 --epoch 60 > logs_moe/none_taowu_decoder328_attrBOLDwin160.log
python trainval.py --dataname neurocon --train_obj sex --decoder --device cuda:0 --epoch 60 > logs_moe/none_neurocon_decoder328_attrBOLDwin160.log
python trainval.py --dataname sz-diana --train_obj sex --decoder --device cuda:0 --epoch 60 > logs_moe/none_sz-diana_decoder328_attrBOLDwin160.log
python trainval.py --dataname abide --train_obj sex --decoder --device cuda:0 --epoch 60 > logs_moe/none_abide_decoder328_attrBOLDwin160.log
