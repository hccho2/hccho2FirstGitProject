# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import threading
import os
from glob import glob
tf.reset_default_graph()

myDataX = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1]]).astype(np.float32)
myDataY = np.array([[0,1,1,1,0,1,1,1,0,1,1,1]]).astype(np.float32).T

log_dir = "hccho-ckpt"
ckpt_file_name_preface = 'model.ckpt'   # 이 이름을 바꾸면, get_most_recent_checkpoint도 바꿔야 한다.
checkpoint_path = os.path.join(log_dir, ckpt_file_name_preface)
def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))

    #latest_checkpoint=checkpoint_paths[0]
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint
class DataFeeder(threading.Thread):
    def __init__(self,sess,coordinator,batch_size):
        super(DataFeeder, self).__init__()
        self.coord = coordinator
        self.batch_size = batch_size
        self.sess = sess
        
        
        self.placeholders = [tf.placeholder(tf.float32, [None,3]), tf.placeholder(tf.float32, [None,1])  ] # 여기에 data를 넣어준다.
        queue = tf.FIFOQueue(capacity=8, dtypes=[tf.float32,tf.float32], name='input_queue')
        self.enqueue_op = queue.enqueue(self.placeholders)
        
        
        # 다음의 self.x, self.y를 모델에서 사용한다.
        self.x, self.y =  queue.dequeue()
        self.x.set_shape(self.placeholders[0].shape)
        self.y.set_shape(self.placeholders[1].shape)
        
    def run(self):
        try:
            while not self.coord.should_stop():
                data_length = len(myDataX)
                
                for _ in range(5): # 필요한 만큼 data를 미리 만들어 놓을 수 있다.
                    s = np.random.choice(data_length,self.batch_size,replace=False)
                    self.sess.run(self.enqueue_op,feed_dict={self.placeholders[0]: myDataX[s],self.placeholders[1]:myDataY[s]})
        except Exception as e:
            print('Exiting due to exception: %s' % e)
            self.coord.request_stop(e)            
            
class SimpleNet():
    # class의 init에서 datafeeder를 받아들이는 것이 좋지 않다. train만 하는 것이 아니고, inference도 있기 때문
    def __init__(self,datafeeder):
        self.datafeeder = datafeeder
        self.build_model()
    def build_model(self):
        
        L1 = tf.layers.dense(self.datafeeder.x,units=4, activation = tf.sigmoid,name='L1')
        L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2')
        self.loss = tf.reduce_mean( 0.5*tf.square(L2-self.datafeeder.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss )


class SimpleNet2():
    # class의 init에서 datafeeder를 받아들이는 것이 좋지 않다. train만 하는 것이 아니고, inference도 있기 때문
    def __init__(self,datafeeder):
        self.datafeeder = datafeeder
        self.build_model()
    def build_model(self):
        
        L1 = tf.layers.dense(self.datafeeder.x,units=4, activation = tf.sigmoid,name='L1')
        L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2')
        self.loss = tf.reduce_mean( 0.5*tf.square(L2-self.datafeeder.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss )



def run_and_save_SimpleNet():
    
    with tf.Session() as sess:
        try:
            coord = tf.train.Coordinator()
            train_feeder = DataFeeder(sess,coord,batch_size=2)
            
            simnet = SimpleNet(train_feeder)  
            step=0
            sess.run(tf.global_variables_initializer())
            train_feeder.start()  # 반드시 있어야됨
            
            
            while not coord.should_stop():
                sess.run(simnet.train_op)
                step = step+1
                 
                if step%1000==0:
                    print("step ",step, ", loss = ", sess.run(simnet.loss))
                
                
                
                if step%30000 ==0:
                    #print(tf.global_variables())
                    saver = tf.train.Saver(tf.global_variables())
                    saver.save(sess, checkpoint_path, global_step=step)

        except Exception as e:
            print('Exiting due to exception: %s' % e)
            coord.request_stop(e)

        finally:
            print('finally')
            coord.request_stop()
            
            
def model_restore_SimpleNet():
    with tf.Session() as sess:
        try:
            coord = tf.train.Coordinator()

            train_feeder = DataFeeder(sess,coord,batch_size=2)
            
            simnet = SimpleNet(train_feeder)  
            sess.run(tf.global_variables_initializer())
            train_feeder.start()
            while not coord.should_stop():
                saver = tf.train.Saver(tf.global_variables())
                
                print(sess.run(tf.get_default_graph().get_tensor_by_name('L1/kernel:0')))
                print('Before restore loss = ', sess.run(simnet.loss))
                
                restore_path = get_most_recent_checkpoint('.//hccho-ckpt')
                saver.restore(sess, restore_path)
                print('model restored!!!')
                #sess.run(tf.global_variables_initializer())  # restore 후, 다시 initializer 하면 안됨.
                print(sess.run(tf.get_default_graph().get_tensor_by_name('L1/kernel:0')))
                print('After restore loss = ', sess.run(simnet.loss))
            
            
                coord.request_stop()
        except Exception as e:
            print('Exiting due to exception: %s' % e)
            coord.request_stop(e)

        finally:
            print('finally')
            coord.request_stop()        
        
if __name__ == '__main__':
    run_and_save_SimpleNet()    
    #model_restore_SimpleNet()