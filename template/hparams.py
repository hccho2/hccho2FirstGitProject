# -*- coding: utf-8 -*-
import tensorflow as tf

hp = tf.contrib.training.HParams(
    log_dir = "hccho-ckpt",
    model_name = "hccho-mm",
    ckpt_file_name_preface = 'model.ckpt',   # 이 이름을 바꾸면, get_most_recent_checkpoint도 바꿔야 한다.
    PARAMS_NAME = "params.json",
    
    learning_rate = 0.01,
    layer_size = [3,1],
)  