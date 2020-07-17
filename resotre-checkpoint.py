import numpy as np
import tensorflow as tf
import collections
import re
tf.reset_default_graph()

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def build_model():
    x = tf.placeholder(tf.float32, shape=(None,5))
    
    L1 = tf.layers.dense(x,6,name='L1')
    L2 = tf.layers.dense(L1,3,name='L2')

    return L2

def pre_train():
    y = build_model()
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    
    saver.save(sess,'saved-model/pre-trained-model')

def checkpoint():
    from tensorflow.contrib.framework.python.framework import checkpoint_utils
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

    checkpoint_filename = 'saved-model/pre-trained-model'
    init_vars = tf.train.list_variables(checkpoint_filename)
    
    for v in init_vars:
        
        vv = checkpoint_utils.load_variable(checkpoint_filename, v[0])
        print(v,vv)

    print('='*20)
    # 모든 변수 출력
    print_tensors_in_checkpoint_file(checkpoint_filename, all_tensors=True, tensor_name='')   
    
    print('='*40)
    # 변수를 골라서 출력.
    for v in init_vars:
        print_tensors_in_checkpoint_file(checkpoint_filename, all_tensors=True, tensor_name=v[0])
    
    
    
    
def fine_tuning():
    checkpoint_filename = 'saved-model/pre-trained-model'
    y = build_model()
    
    pretrained_vars = tf.trainable_variables()
    
    (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(pretrained_vars, checkpoint_filename)
    # assignment_map: OrderedDict([('L1/bias', 'L1/bias'), ('L1/kernel', 'L1/kernel'), ('L2/bias', 'L2/bias'), ('L2/kernel', 'L2/kernel')])
    # initialized_variable_names: {'L1/bias': 1, 'L1/bias:0': 1, 'L1/kernel': 1, 'L1/kernel:0': 1, 'L2/bias': 1, 'L2/bias:0': 1, 'L2/kernel': 1, 'L2/kernel:0': 1}
    
    
    # tf.train.init_from_checkpoint를 통해, tf.global_variables_initializer()가 실행될 때, random 초기화가 되지 않는다.
    tf.train.init_from_checkpoint(checkpoint_filename, assignment_map)  # 여기서 실행되는 것은 아니다. 변수 초기화가 될 때, 실행된다.
    
    ########################################################
    # fine tuning model
    z = tf.layers.dense(y,1,name='L3')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    all_tvars = tf.trainable_variables()
    
    
    for v in all_tvars:
        value = sess.run(v)
        print(v.name, value)

if __name__ == '__main__':
    #pre_train()  # checkpoint를 하나 만든다.
    checkpoint()  # pre trained model로 저장한 checkpoint의 값을 확인한다.
    #fine_tuning()










