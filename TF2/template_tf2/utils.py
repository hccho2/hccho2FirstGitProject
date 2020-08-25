# -*- coding: utf-8 -*-
import os
import json
from glob import glob
from datetime import datetime
from collections import namedtuple
import infolog
def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
    
    if checkpoint_paths == []: 
        return ''
    
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))

    #latest_checkpoint=checkpoint_paths[0]
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint

def prepare_dirs(hp, load_path=None):
    # load_path = 'hccho-ckpt\\hccho-mm-2019-07-31_13-56-59'
    # checkpoint_path = 'hccho-ckpt\\hccho-mm-2019-08-02_10-21-12\\model.ckpt'
    # restore_path = 'hccho-ckpt\\hccho-mm-2019-08-02_09-56-45\\model.ckpt-120000'
    
    from shutil import copyfile as copy_file
    
    def get_time():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    
    if load_path is None:
        load_path = os.path.join(hp['log_dir'], "{}-{}".format(hp['model_name'], get_time()))
        os.makedirs(load_path)
        
        save_hparams(load_path, hp)  # hp에 있는 내용을 json파일로 저장한다.
        copy_file(hp['hp_filename'], os.path.join(load_path, hp['hp_filename']))
        
    else:
        load_hparams(hp, load_path)
        
    checkpoint_path = os.path.join(load_path, hp['ckpt_file_name_preface'])
    restore_path = get_most_recent_checkpoint(load_path)


    
    return load_path,restore_path,checkpoint_path


def load_hparams(hparams, load_path, skip_list=[]):
    # log dir에 있는 hypermarameter 정보를 이용해서, hparams.py의 정보를 update한다.
    path = os.path.join(load_path, hparams['PARAMS_NAME'])

    new_hparams = load_json(path)
    hparams_keys =hparams.keys()

    for key, value in new_hparams.items():
        if key in skip_list or key not in hparams_keys:
            print("Skip {} because it not exists".format(key))  #json에 있지만, hparams에 없다는 의미
            continue

        if key not in ['xxxxx',]:  # update 하지 말아야 할 것을 지정할 수 있다.
            original_value = hparams.get(key)
            if original_value != value:
                print("UPDATE {}: {} -> {}".format(key, hparams.get(key), value))
                hparams[key] = value

def save_hparams(model_dir, hparams):
    
    param_path = os.path.join(model_dir, hparams['PARAMS_NAME'])
    # hparams.to_json() ---> string return ---> '{"log_dir": "hccho-ckpt", "model_name": "hccho-mm", "ckpt_file_name_preface": "model.ckpt", "PARAMS_NAME": "params.json", "learning_rate": 0.01, "layer_size": [3, 1]}'
    
    #info = eval(hparams.to_json(),{'false': False, 'true': True, 'null': None})   # python의 eval함수가 dict형으로 변환해 준다.
    info = dict(hparams)
    
    write_json(param_path, info)

    print(" [*] MODEL dir: {}".format(model_dir))
    print(" [*] PARAM path: {}".format(param_path))

def load_json(path, as_class=False, encoding='euc-kr'):
    import re
    with open(path,encoding=encoding) as f:
        content = f.read()
        content = re.sub(",\s*}", "}", content)  # ,공백}  ---> }
        content = re.sub(",\s*]", "]", content)

        if as_class:
            data = json.loads(content, object_hook=\
                    lambda data: namedtuple('Data', data.keys())(*data.values()))
        else:
            data = json.loads(content)  # string --> dict

    return data   
def write_json(path, data):
    with open(path, 'w',encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)


def prepare(hp,load_path):
    load_path,restore_path,checkpoint_path = prepare_dirs(hp,load_path)
    log = infolog.log
    log_path = os.path.join(load_path, 'train.log')
    
    if 'slack_token' in hp.keys():
        slack_token = hp['slack_token']
        infolog.init(log_path, hp['model_name'],slack_token)
    else:
        infolog.init(log_path, hp['model_name'])
    
    return log, load_path,restore_path,checkpoint_path
