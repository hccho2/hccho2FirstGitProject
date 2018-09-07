# coding: utf-8

import tensorflow as tf
import numpy as np
import threading
myDataX = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1]]).astype(np.float32)
myDataY = np.array([[0,1,1,1,0,1,1,1,0,1,1,1]]).astype(np.float32).T


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
        
    def run(self):  # run이 반드시 정의되어야 한다.
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
    def __init__(self,datafeeder):
        self.datafeeder = datafeeder
        self.build_model()
    def build_model(self):
        
        L1 = tf.layers.dense(self.datafeeder.x,units=4, activation = tf.sigmoid,name='L1')
        L2 = tf.layers.dense(L1,units=1, activation = tf.sigmoid,name='L2')
        self.loss = tf.reduce_mean( 0.5*tf.square(L2-self.datafeeder.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss )
    
    

def main():
    with tf.Session() as sess:
        try:
            coord = tf.train.Coordinator()
            train_feeder = DataFeeder(sess,coord,batch_size=2)
            
            simnet = SimpleNet(train_feeder)  
            i=0
            sess.run(tf.global_variables_initializer())
            train_feeder.start()  # 반드시 있어야됨
            
            
            while not coord.should_stop():
                sess.run(simnet.train_op)
                i = i+1
                 
                if i%1000==0:
                    print(sess.run(simnet.loss))
                    
                if i>=8000:
                    coord.request_stop()
            """    
            for step in range(3000):
                sess.run(simnet.train_op)
                if step%1000==0:
                    print(sess.run(simnet.loss))                
            """ 
        except Exception as e:
            print('Exiting due to exception: %s' % e)
            coord.request_stop(e)

        finally:
            print('finally')
            coord.request_stop()
         

if __name__ == '__main__':
    main()




