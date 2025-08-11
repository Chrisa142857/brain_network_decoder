## Codes for the large connectome model (LCM).

The official implementation of paper "Large Connectome Model: An fMRI Foundation Model of Brain Connectomes Empowered by Brain-Environment Interaction in Multitask Learning Landscape". Pre-trained model weights can be found here: [https://drive.google.com/drive/folders/1vEIB6qb5djrQXLuRYeLrkF2i27m4QNgo?usp=sharing](https://drive.google.com/drive/folders/1vEIB6qb5djrQXLuRYeLrkF2i27m4QNgo?usp=sharing).

### File structure

```
├── data                                   # Will be created when pre-loading data
├── datasets.py                            # The class of datasets and dataloaders
├── decoder_vs_mlp_in_markdown.md          # A markdown table parsed from log files of exp
├── finetune_lbnm.py                       # Finetune LCM
├── get_sex_age_score.py                   # Parse sex and age score in log files
├── lineplot_modelScale.py                 # Line plots for model scalability
├── logs                                   # Will be created when running exp
├── md2latex.py                            # Convert markdown table to Latex
├── models                             
│   ├── BNT                                # BNT codes migrated from the official github repo
│   ├── BolT                               # BolT codes migrated from the official github repo
│   ├── bolt.py                            # Build BolT model
│   ├── brain_gnn.py                       # BrainGNN codes migrated from the official github repo
│   ├── brain_identity.py                  # Linear model
│   ├── brain_net_transformer.py           # Build BNT model
│   ├── fbnetgen.py                        # Not used
│   ├── graphormer.py                      # Graphormer codes migrated from the official github repo
│   ├── heads.py                           # Decoder including **LCM**
│   ├── hunet.py                           # Not used
│   ├── nagphormer.py                      # NAGphormer codes migrated from the official github repo
│   ├── neuro_detour.py                    # NeuroPath codes migrated from the official github repo
│   ├── PathNN.py                          # Not used
│   ├── utils.py                           # Util codes for model building
│   └── vanilla_model.py                   # Not used
├── model_weights                          # Put folders of model weights under this dir
├── parse_log_decoderVSmlp.py              # Parse log files of exp as a markdown table
├── plot_param_vs_layern.py                # Not used
├── plot_surfIce.py                        # Scripts for brain surface attention visualization on software *Surf Ice*
├── print_model_scale.py                   # Show model parameter number
├── save_attention_map.py                  # Extract cross-attention map of LCM
├── scripts                            
│   ├── exp_script_finetine.sh             # Finetuning experiments
│   ├── exp_script_finetune2.sh            # Finetuning experiments
│   ├── exp_script_jiaxingData.sh          # Part of experiments in Table 1, 3, 4
│   ├── exp_script_LBNM.sh                 # Part of experiments in Table 1, 3, 4
│   ├── exp_script_modelScal_decoder.sh    # Part of experiments in Figure 6 and Table 2
│   ├── exp_script_modelScale.sh           # Part of experiments in Figure 6 and Table 2
│   ├── exp_script_modelScale_tesla.sh     # Part of experiments in Figure 6 and Table 2
│   ├── exp_script_multiBackbone.sh        # Part of experiments in Table 1, 3, 4
│   ├── exp_script_multiBackbone_tesla.sh  # Part of experiments in Table 1, 3, 4
│   └── exp_scripts.sh                     # Part of experiments
├── trainval_lbnm.py                       # Pre-train LCM with multiple datasets
└── trainval.py                            # Train models with single dataset
```

