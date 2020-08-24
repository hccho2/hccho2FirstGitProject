# -*- coding: utf-8 -*-
from hparams import HParams   # pip install hparams


hp = HParams(
    log_dir = "hccho-ckpt",
    model_name = "hccho-mm",  # log_dir + model_name + 날짜시간   --> hccho-ckpt\hccho-mm-2020-08-24_17-08-31
    ckpt_file_name_preface = 'model.ckpt',   # checkpoint preface
    PARAMS_NAME = "params.json",
    hp_filename = 'hyper_params.py',  # 이 파일 자체의 이름
    
    learning_rate = 0.02,
    layer_size = [3,1],
    
    num_epoch = 10,
    batch_size = 2,
)