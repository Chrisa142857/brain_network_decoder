

python zsl_lbnm.py --dataname ppmi --device cuda:1 > logs3/noneKmeans_ppmi2cls_2datasetDecoder32_attrFC.log
python zsl_lbnm.py --dataname adni --device cuda:1 > logs3/noneKmeans_adni_2datasetDecoder32_attrFC.log
python zsl_lbnm.py --dataname abide --device cuda:1 > logs3/noneKmeans_abide_2datasetDecoder32_attrFC.log
# python zsl_lbnm.py --dataname taowu --cv_fold_n 5 --device cuda:1 > logs3/noneKmeans_taowu_2datasetDecoder32_attrFC.log
# python zsl_lbnm.py --dataname neurocon --cv_fold_n 5 --device cuda:1 > logs3/noneKmeans_neurocon_2datasetDecoder32_attrFC.log

python zsl_lbnm.py --dataname sz-diana --device cuda:1 --few_shot 0.01 > logs3/noneGridsearchFewshot001_sz-diana_2datasetDecoder32_attrFC.log
python zsl_lbnm.py --dataname ppmi --device cuda:1 --few_shot 0.01 > logs3/noneGridsearchFewshot001_ppmi2cls_2datasetDecoder32_attrFC.log
python zsl_lbnm.py --dataname abide --device cuda:1 --few_shot 0.01 > logs3/noneGridsearchFewshot001_abide_2datasetDecoder32_attrFC.log