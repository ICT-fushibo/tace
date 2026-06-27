Finetune Tutorial
=================

This section introduces how to finetune **TACE** models.

We provide a variety of pretrained TACE foundation models. You can finetune these
foundation models directly, or finetune your own models.

Currently, TACE supports three major finetuning strategies:

1. Full parameters 
2. Freeze parameters
3. Low-Rank Adaptation (LoRA)


Motivation for Finetuning
-------------------------

The main goal of finetuning is to **preserve as much knowledge as possible from the
foundation model**, while achieving high accuracy on your target task.

Therefore,  when finetuning dataset is relatively small, full-parameter finetuning is discouraged. 
Full parameters finetuning may lead to:

- Overfitting due to limited training data
- Catastrophic forgetting of knowledge learned during pretraining

To mitigate these issues, parameter-efficient finetuning methods such as LoRA and
freezing pretrained parameters are generally preferred.

Finetuning Strategies
---------------------

Below we describe the supported finetuning strategies and how to use them in practice.

Freezing Pretrained Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Freezing pretrained parameters is not discussed in detail here. In the finetuning
configuration files that we automatically generate, **all pretrained parameters are
frozen by default**.

This design choice helps preserve the knowledge learned during pretraining and avoids
overfitting when the finetuning dataset is limited.

Low-Rank Adaptation (LoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^

In short, LoRA introduces additional trainable parameters on top of an existing model,
according to user-specified configurations. These trainable parameters follow the same
structural pattern as the original model layers.

After training, the LoRA weights are **merged into the base model weights**, so the final
exported model does **not introduce any additional parameters** compared to the original
model.

LoRA is mainly controlled by two key hyperparameters:

- ``rank`` (int): determines the number of additional trainable parameters introduced
  during LoRA finetuning.
- ``alpha`` (float): controls the strength of the LoRA update.

The ``rank`` parameter is typically in the range of **4 to 32**. A larger
rank increases the number of trainable LoRA parameters.

The ``alpha`` parameter is commonly set within the range of **r to 2r**, where ``r`` denotes the
LoRA rank.

Replay Data
^^^^^^^^^^^

During finetuning, it is possible to mix in a portion of the training data used for
the foundation model in order to mitigate catastrophic forgetting.

However, in our current version, we do not include replay data such as
multi-fidelity or multi-head training data during finetuning by default.


Example
-------

First, download a pretrained model from
https://github.com/xvzemin/tace-foundations
(you may also use a model pretrained by yourself).

You can then follow the example below, which consists of three main steps:

1. Before training, use ``tace-finetune`` to automatically generate a finetuning
   configuration file named ``finetune_config.yaml``. You can modify this file
   to adjust the desired finetuning parameters.

2. After preparing your training config, start the finetuning process using
   ``tace-train``. TACE will automatically load and apply the settings from
   ``finetune_config.yaml``.

3. After training is completed, use ``tace-convert`` to convert the generated
   LoRA checkpoint (``*.ckpt`` file) into a standard model by merging the LoRA
   weights into the base model. The resulting model can then be deployed for
   production use.

Example commands are shown below:

.. code-block:: bash

   tace-finetune -m TACE-OAM-RRA.pt

   # Start finetuning (configuration file specified as needed)
   tace-train -cn *.yaml

   # Merge LoRA weights into the base model
   tace-convert -m *.ckpt --type merge_lora
