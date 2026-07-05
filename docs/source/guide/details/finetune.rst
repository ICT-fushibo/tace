finetune_from_model
===================

The ``finetune_from_model`` field does not contain any additional parameters.  
You only need to provide the path to a ``.ckpt``, ``.pt``, or ``.pth`` file to perform finetuning.  
If finetuning is not required, this field should be set to ``null``.

Additionally, if a ``finetune_config.yaml`` file exists in the current directory, 
the model will default to LoRA finetuning. If this file does not exist, full-model 
finetuning will be applied. Use ``tace-finetune --model `` to generate finetune config.
