# LauraGPT


Unofficial Implementation for [LauraGPT](https://arxiv.org/abs/2310.04673)

This repository simplifies the code for training the LauraGPT model.

Prerequisite: 
    1. FunCodec Installed

We currently support
    - [x] LauraTTS.


To run the training scheme, do 
```shell
python3 train.py --config config/conf.yaml
```

It will train on gpus 0,1,2,3. 

We use 6 layers of the decoder-only LM instead of 12 as proposed in the config lauraGPT model.

We trained our model on 32GB GPU cards.
