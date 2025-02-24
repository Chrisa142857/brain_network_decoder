
# python trainval.py --force_2class --few_shot 0.5 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot05_ppmi2cls_mlp1_attrFC.log &
# pid1=$!
# python trainval.py --force_2class --few_shot 0.5 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot05_ppmi2cls_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2
# # python trainval.py --force_2class --few_shot 0.5 --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot05_ppmi2cls_mlp1_attrFC.log  &
# pid1=$!
# # python trainval.py --force_2class --few_shot 0.5 --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot05_ppmi2cls_mlp1_attrFC.log  &
# # python trainval.py --force_2class --few_shot 0.5 --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot05_ppmi2cls_mlp1_attrFC.log  &
# python trainval.py --force_2class --few_shot 0.5 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot05_ppmi2cls_mlp1_attrFC.log  &
# pid2=$!
# wait $pid1
# wait $pid2

# python trainval.py --force_2class --few_shot 0.3 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot03_ppmi2cls_mlp1_attrFC.log  &
# pid1=$!
# python trainval.py --force_2class --few_shot 0.3 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot03_ppmi2cls_mlp1_attrFC.log  &
# pid2=$!
# wait $pid1
# wait $pid2
# # python trainval.py --force_2class --few_shot 0.3 --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot03_ppmi2cls_mlp1_attrFC.log  &
# pid1=$!
# # python trainval.py --force_2class --few_shot 0.3 --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot03_ppmi2cls_mlp1_attrFC.log  &
# # python trainval.py --force_2class --few_shot 0.3 --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot03_ppmi2cls_mlp1_attrFC.log  &
# python trainval.py --force_2class --few_shot 0.3 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot03_ppmi2cls_mlp1_attrFC.log  &
# pid2=$!
# wait $pid1
# wait $pid2

# python trainval.py --force_2class --few_shot 0.1 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot01_ppmi2cls_mlp1_attrFC.log  &
# pid1=$!
# python trainval.py --force_2class --few_shot 0.1 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot01_ppmi2cls_mlp1_attrFC.log  &
# pid2=$!
# wait $pid1
# wait $pid2
# # python trainval.py --force_2class --few_shot 0.1 --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot01_ppmi2cls_mlp1_attrFC.log &
# pid1=$!
# # python trainval.py --force_2class --few_shot 0.1 --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot01_ppmi2cls_mlp1_attrFC.log  &
# # python trainval.py --force_2class --few_shot 0.1 --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot01_ppmi2cls_mlp1_attrFC.log  &
# python trainval.py --force_2class --few_shot 0.1 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot01_ppmi2cls_mlp1_attrFC.log &
# pid2=$!
# wait $pid1
# wait $pid2

# python trainval.py --force_2class --few_shot 0.01 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot001_ppmi2cls_mlp1_attrFC.log  &
# pid1=$!
# python trainval.py --force_2class --few_shot 0.01 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot001_ppmi2cls_mlp1_attrFC.log  &
# pid2=$!
# wait $pid1
# wait $pid2
# # python trainval.py --force_2class --few_shot 0.01 --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot001_ppmi2cls_mlp1_attrFC.log  &
# pid1=$!
# # python trainval.py --force_2class --few_shot 0.01 --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot001_ppmi2cls_mlp1_attrFC.log  &
# # python trainval.py --force_2class --few_shot 0.01 --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot001_ppmi2cls_mlp1_attrFC.log  &
# python trainval.py --force_2class --few_shot 0.01 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot001_ppmi2cls_mlp1_attrFC.log  &
# pid2=$!
# wait $pid1
# wait $pid2

# python trainval.py --force_2class --few_shot 1 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShot10_ppmi2cls_mlp1_attrFC.log  &
# pid1=$!
# python trainval.py --force_2class --few_shot 1 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShot10_ppmi2cls_mlp1_attrFC.log  &
# pid2=$!
# wait $pid1
# wait $pid2
# # python trainval.py --force_2class --few_shot 1 --models braingnn --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:0 --cv_fold_n 5 --batch_size 128 > logs3/braingnnFewShot10_ppmi2cls_mlp1_attrFC.log  &
# # python trainval.py --force_2class --few_shot 1 --models graphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/graphormerFewShot10_ppmi2cls_mlp1_attrFC.log  &
# # python trainval.py --force_2class --few_shot 1 --models nagphormer --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:2 --cv_fold_n 5 --batch_size 128 > logs3/nagphormerFewShot10_ppmi2cls_mlp1_attrFC.log  &
# python trainval.py --force_2class --few_shot 1 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShot10_ppmi2cls_mlp1_attrFC.log 



python trainval.py --force_2class --few_shot 0.01 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv2001_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 0.01 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv2001_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 0.01 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv2001_ppmi2cls_mlp1_attrFC.log  

python trainval.py --force_2class --few_shot 0.1 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv201_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 0.1 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv201_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 0.1 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv201_ppmi2cls_mlp1_attrFC.log  

python trainval.py --force_2class --few_shot 0.3 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv203_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 0.3 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv203_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 0.3 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv203_ppmi2cls_mlp1_attrFC.log  

python trainval.py --force_2class --few_shot 0.5 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv205_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 0.5 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv205_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 0.5 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv205_ppmi2cls_mlp1_attrFC.log  

python trainval.py --force_2class --few_shot 1 --models bolt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/boltFewShotv210_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 1 --models bnt --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/bntFewShotv210_ppmi2cls_mlp1_attrFC.log  
python trainval.py --force_2class --few_shot 1 --models neurodetour --dataname ppmi --node_attr FC --decoder_layer 1 --device cuda:1 --cv_fold_n 5 --batch_size 128 > logs3/neurodetourFewShotv210_ppmi2cls_mlp1_attrFC.log  
