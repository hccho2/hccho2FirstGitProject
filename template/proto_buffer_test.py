# -*- coding: utf-8 -*-
'''
1. pbtxt 파일을 만들어, 모델 구조를 저장한다.
2. freeze_graph()를 이용하여, weight와 모델 구조가 모두 저장되는 pb파일을 만든다.


'''
import tensorflow as tf
import numpy as np
from tensorflow.python.tools import freeze_graph
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y = np.array([[0,1,1,1]]).T


def make_pb():


    x = tf.placeholder(tf.float32, [None,3])
    y = tf.placeholder(tf.float32, [None,1])
    L1 = tf.layers.dense(x,units=4, activation = tf.nn.relu,name='L1')
    L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2')
    train = tf.train.AdamOptimizer(learning_rate=0.01).minimize( tf.reduce_mean( 0.5*tf.square(L2-y)))
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for j in range(1000):
        sess.run(train, feed_dict={x: X, y: Y})
    
    print(sess.run(L2,feed_dict={x: X}))
    
    saver = tf.train.Saver()
    saver.save(sess,'./model_pb/model.ckpt')
    
    # 모델 구조만 나간다.
    tf.train.write_graph( sess.graph, "./model_pb", "my_graph.pbtxt", as_text=True )  # text 파일이므로, 읽을 수는 있지만, 파일 사이즈가 크다.


    '''
    > python freeze_graph.py 
    --input_graph = ./model_pb/my_graph.pbtxt --input_checkpoint = ./model_pb/model.ckpt --output_graph=./model_pb/my_graph.pb
    
    
    '''
def freeze():
    input_graph = './model_pb/my_graph.pbtxt'
    input_checkpoint = './model_pb/model.ckpt'
    output_graph = './model_pb/my_graph.pb'
    output_node_names = 'L2/Relu'   # 어디까지 내 보낼지 

    freeze_graph.freeze_graph(input_graph,"", False,input_checkpoint,output_node_names,None,None,output_graph,True,None)
    
    
def load_pb():
    pb_path = "./model_pb/my_graph.pb"
    
    my_graph = tf.Graph()
    with my_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    print([n.name for n in my_graph.as_graph_def().node])  # 15개


    with my_graph.as_default():
        in_tensor = my_graph.get_tensor_by_name('Placeholder:0')
        out_tensor = my_graph.get_tensor_by_name('L2/Sigmoid:0')
        
        w = my_graph.get_tensor_by_name('L2/kernel:0')
        
        with tf.Session(graph=my_graph) as sess:

            print(sess.run(out_tensor, feed_dict={in_tensor: X}))
            print(sess.run(w))

    print('Done')



if __name__ == '__main__':
    #make_pb()
    #freeze()
    load_pb()