

python trainval.py --few_shot 0.01 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv2001_abide_mlp1_attrFC.log  
python trainval.py --few_shot 0.01 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv2001_abide_mlp1_attrFC.log  
python trainval.py --few_shot 0.01 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv2001_abide_mlp1_attrFC.log  

python trainval.py --few_shot 0.1 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv201_abide_mlp1_attrFC.log  
python trainval.py --few_shot 0.1 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv201_abide_mlp1_attrFC.log  
python trainval.py --few_shot 0.1 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv201_abide_mlp1_attrFC.log  

python trainval.py --few_shot 0.3 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv203_abide_mlp1_attrFC.log  
python trainval.py --few_shot 0.3 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv203_abide_mlp1_attrFC.log  
python trainval.py --few_shot 0.3 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv203_abide_mlp1_attrFC.log  

python trainval.py --few_shot 0.5 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv205_abide_mlp1_attrFC.log  
python trainval.py --few_shot 0.5 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv205_abide_mlp1_attrFC.log  
python trainval.py --few_shot 0.5 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv205_abide_mlp1_attrFC.log  

python trainval.py --few_shot 1 --models bolt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv210_abide_mlp1_attrFC.log  
python trainval.py --few_shot 1 --models bnt --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv210_abide_mlp1_attrFC.log  
python trainval.py --few_shot 1 --models neurodetour --dataname abide --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv210_abide_mlp1_attrFC.log  
