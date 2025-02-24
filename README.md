## Codes for the large connectome model (LCM).

The official implementation of paper "Large Connectome Model: An fMRI Foundation Model of Brain Connectomes Powered by Brain-Environment Interactions".

### File structure

```
├── data                                   # Will be created when pre-loading data
├── datasets.py                            # The class of datasets and dataloaders
├── logs3_in_markdown.md                   # A markdown table of experimental results parsed from log files
├── finetune_lbnm.py                       # Finetune LCM
├── logs3                                  # Will be created when running exp
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
├── parse_logs3.py                         # Parse log files of exp as a markdown table
├── print_model_scale.py                   # Show model parameter number
├── save_attention_map.py                  # Extract BECA map
├── scripts                            
│   ├── exp_script_zeroshot.sh                  # Zero shot experiments    
│   ├── exp_script_fewshot_randomDecoder.sh     # Few shot experiments
│   ├── exp_script_fewshot_finetune.sh          # Few shot experiments
│   ├── exp_script_fewshot_finetune2.sh         # Few shot experiments
│   ├── exp_script_fewshot.sh                   # Few shot experiments
│   ├── exp_script_finetune_baseline2.sh        # Few shot experiments
│   ├── exp_script_ppmi2cls.sh                  # Experiments on PPMI
│   └── exp_script_ppmi2cls_2.sh                # Experiments on PPMI
├── zsl_lbnm.py                            # Zero shot learning LCM
├── trainval_lbnm.py                       # Pre-train LCM with multiple datasets
└── trainval.py                            # Train models with single dataset
```

