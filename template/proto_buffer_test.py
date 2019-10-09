# -*- coding: utf-8 -*-
'''
1. pbtxt 파일을 만들어, 모델 구조를 저장한다.
2. freeze_graph()를 이용하여, weight와 모델 구조가 모두 저장되는 pb파일을 만든다.

또는 graph_util.convert_variables_to_constants를 이용해서, weight값과 구조가 모두 저장되는 pb파일을 한번에 만들 수도 있다.

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
    
    
    # 어떤 tensor가 있는지 확인
    all_tensor =[n.name for n in sess.graph.as_graph_def().node]
    print(all_tensor)
    
    # 모델 구조만 나간다. ---> freeze과정을 추가로 거쳐야 한다.
    tf.train.write_graph( sess.graph, "./model_pb", "my_graph.pbtxt", as_text=True )  # text 파일이므로, 읽을 수는 있지만, 파일 사이즈가 크다.



    ##
    from tensorflow.python.framework import graph_util
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['L2/Sigmoid'])
    with  tf.gfile.GFile("./model_pb/my_graph2.pb", 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def freeze():
    '''
    cmd창에서도 할 수 있다.
    > python freeze_graph.py 
    --input_graph = ./model_pb/my_graph.pbtxt --input_checkpoint = ./model_pb/model.ckpt --output_graph=./model_pb/my_graph.pb
    
    '''    

    
    input_graph = './model_pb/my_graph.pbtxt'
    input_checkpoint = './model_pb/model.ckpt'
    output_graph = './model_pb/my_graph.pb'  # 저장할 파일 이름
    output_node_names = 'L2/Sigmoid'   # 어디까지 내 보낼지 

    freeze_graph.freeze_graph(input_graph,"", False,input_checkpoint,output_node_names,None,None,output_graph,True,None)
    





def load_pb():
    # my_graph.pb, my_graph2.pb 두개 모두 가능.
    #pb_path = "./model_pb/my_graph.pb"
    pb_path = "./model_pb/my_graph2.pb"
    
    my_graph = tf.Graph()
    with my_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            
            #return_elements = None
            return_elements = ['L1/Relu:0','L2/Sigmoid:0']  # return_elements을 통해 tensor를 뽑아낼 수도 있고, get_tensor_by_name()을 통해서 할 수도 있다.
            a = tf.import_graph_def(od_graph_def, name='',return_elements=return_elements)
            
    print([n.name for n in my_graph.as_graph_def().node])  # 15개


    with my_graph.as_default():
        in_tensor = my_graph.get_tensor_by_name('Placeholder:0')
        out_tensor = my_graph.get_tensor_by_name('L2/Sigmoid:0')
        
        w = my_graph.get_tensor_by_name('L2/kernel:0')
        
        with tf.Session(graph=my_graph) as sess:
            print('tf.trainable_variables()', tf.trainable_variables())
            sess.run(tf.global_variables_initializer())  # pb파일로 부터 생성된 tensor는 trainable이 아니다.

            print('target: ',Y, 'prediction: ', sess.run(out_tensor, feed_dict={in_tensor: X}))
            print('w: ',sess.run(w))


    with my_graph.as_default():
        # graph로 부터 뽑아낸 tensor른 placeholder 역할을 할 수도 있다.  <----- 이는 특별한 것이 아니고, 원래 placeholder가 아니어도 되는 것이다.
        middle_tensor = my_graph.get_tensor_by_name('L1/Relu:0')
        
        with tf.Session(graph=my_graph) as sess:
            m =sess.run(middle_tensor, feed_dict={in_tensor: X})
            print('L1: ', m)
            
            y = sess.run(out_tensor, feed_dict={middle_tensor: m})
            print('prediction: ', y)
    
    
    with my_graph.as_default():
        all_tensor =[my_graph.get_tensor_by_name(n.name+':0') for n in my_graph.as_graph_def().node]
        print(all_tensor)
        for t in all_tensor:
            print(t, my_graph.is_feedable(t)) # 모든 tensor가 feedable하다.

    print('Done')
    
    
    
    



if __name__ == '__main__':
    #make_pb()
    #freeze()
    load_pb()