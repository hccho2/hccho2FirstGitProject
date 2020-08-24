# -*- coding: utf-8 -*-
from hparams import HParams   # pip install hparams


hp = HParams(
    log_dir = "hccho-ckpt",
    model_name = "hccho-mm",
    ckpt_file_name_preface = 'model.ckpt',   # 이 이름을 바꾸면, get_most_recent_checkpoint도 바꿔야 한다.
    PARAMS_NAME = "params.json",
    hp_filename = 'hyper_params.py',  # 이 파일 자체의 이름
    
    learning_rate = 0.02,
    layer_size = [3,1],
    
    num_epoch = 10,
    batch_size = 2,
)