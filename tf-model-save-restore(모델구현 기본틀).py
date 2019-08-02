# -*- coding: utf-8 -*-
"""
run_and_save_SimpleNet() ---> DataFeeder(Queue) + Model class(SimpleNet) ---> 이 방식은 train만 고려 --> 좋지 못함.
run_and_save_SimpleNet2() ---> DataFeeder(Queue) + Model class(SimpleNet2) --> train, inference 모두 고려 ----> 나의 표준 방식
run_and_save_SimpleNet3() --->  DataFeeder2(tf.data) + Model class(SimpleNet2) + tf.estimator  ----> 이 방식도 좋음. 단 data전체가 아니라, mini-batch data의 조작이 불가.


run_and_save_SimpleNet4()  ---> DataFeeder(Queue) + Model class  + tf.estimator  ---> 이게 가능한다. 
tf.estimator는 Session이 숨겨져 있고, DataFeeder(Queue)는 Session이 필요하다. 
placeholder를 중간에 끼는 것으로  성공.  ----> 좀 느리다.



"""

"""
먼저, model_name, log_dir, ckpt_file_name_preface가 주어져 있어야 된다.
model_name = "hccho-mm"
log_dir = "hccho-ckpt"    # 'logs-hccho'
ckpt_file_name_preface = 'model.ckpt'   # 이 이름을 바꾸면, get_most_recent_checkpoint도 바꿔야 한다.


#load_path = None  # 새로운 training
load_path = 'hccho-ckpt\\hccho-mm-2019-07-31_13-56-59'

1. load_path 설명. None이거나 주어져 있거나(user가 설정해야 됨). None이면 log_dir를 이용해서 새로 만든다.
2. lad_path(주어져 있거나, 새로 만들거나 뭔가 주어져 있다) --> restore_path = get_most_recent_checkpoint(log_dir)  e.g. restore_path = 'hccho-ckpt\\hccho-mm-2019-08-02_09-56-45\\model.ckpt-120000'
3. checkpoint_path  ---> 'hccho-ckpt\\hccho-mm-2019-08-02_10-21-12\\model.ckpt'



3. 참고로 estimator의 model_dir 는 load_path와 동등함.


"""




import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import threading
import os
from glob import glob
from datetime import datetime
tf.reset_default_graph()

 

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
    
    def get_time():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if load_path is None:
        load_path = os.path.join(hp.log_dir, "{}-{}".format(hp.model_name, get_time()))
        os.makedirs(load_path)
        
    checkpoint_path = os.path.join(load_path, hp.ckpt_file_name_preface)
    restore_path = get_most_recent_checkpoint(load_path)
    
    return load_path,restore_path,checkpoint_path,

myDataX = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1],[0,0,1],[0,1,1],[1,0,1],[1,1,1]]).astype(np.float32)
myDataY = np.array([[0,1,1,1,0,1,1,1,0,1,1,1]]).astype(np.float32).T




hp = tf.contrib.training.HParams(
    log_dir = "hccho-ckpt",
    model_name = "hccho-mm",
    ckpt_file_name_preface = 'model.ckpt',   # 이 이름을 바꾸면, get_most_recent_checkpoint도 바꿔야 한다.
    
    
    learning_rate = 0.1,
    layer_size = [3,1],
)  

#load_path = None  # 새로운 training
load_path = 'hccho-ckpt\\hccho-mm-2019-08-02_11-00-23'
#####


load_path,restore_path,checkpoint_path = prepare_dirs(hp,load_path)
print("checkpoint_path: ", checkpoint_path) # hccho-ckpt\hccho-mm-2019-07-31_13-56-59\model.ckpt




class DataFeeder(threading.Thread):
    def __init__(self,coordinator,batch_size):
        super(DataFeeder, self).__init__()
        self.coord = coordinator
        self.batch_size = batch_size
        self.step = 0  # train step에 따라 data feed 성격이 달라 질 것에 대비
        
        
        self.placeholders = [tf.placeholder(tf.float32, [None,3]), tf.placeholder(tf.float32, [None,1])  ] # 여기에 data를 넣어준다.
        queue = tf.FIFOQueue(capacity=8, dtypes=[tf.float32,tf.float32], name='input_queue')
        self.enqueue_op = queue.enqueue(self.placeholders)
        
        
        # 다음의 self.x, self.y를 모델에서 사용한다.
        self.x, self.y =  queue.dequeue()
        self.x.set_shape(self.placeholders[0].shape)
        self.y.set_shape(self.placeholders[1].shape)


    def start_in_session(self, session, start_step):
        self.step = start_step
        self.sess = session
        self.start()
        
    def run(self):
        # 여기서 mini batch를 만드는 op를 sess.run하면 된다.
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
        
    def add_optimizer(self,global_step):
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss,global_step=global_step )
        return self.train_op




def run_and_save_SimpleNet():
    
    coord = tf.train.Coordinator()
    train_feeder = DataFeeder(coord,batch_size=2)
    simnet = SimpleNet(train_feeder)  
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = simnet.add_optimizer(global_step)
    
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        try:
            
            sess.run(tf.global_variables_initializer())
            
            # 모델 restore
            if restore_path == '':
                start_step=0
                sess.run(tf.assign(global_step, 0))
            else:
                saver.restore(sess, restore_path)
                print('Resuming from checkpoint: %s' %(restore_path))
            
            start_step = sess.run(global_step)
            
            
            train_feeder.start_in_session(sess,start_step)  # 반드시 있어야됨
            while not coord.should_stop():
                step, _ =sess.run([global_step,train_op])
                 
                if step%1000==0:
                    print("step ",step, ": loss = ", sess.run(simnet.loss))
                
                
                
                if step%30000 ==0:
                    #print(tf.global_variables())
                    
                    saver.save(sess, checkpoint_path, global_step=step)

        except Exception as e:
            print('Exiting due to exception: %s' % e)
            coord.request_stop(e)

        finally:
            print('finally')
            coord.request_stop()
            
            
def model_restore_SimpleNet():
    
    coord = tf.train.Coordinator()

    train_feeder = DataFeeder(coord,batch_size=2)   
    start_step = 0
    with tf.Session() as sess:
        try:

            
            simnet = SimpleNet(train_feeder)  
            sess.run(tf.global_variables_initializer())
            train_feeder.start_in_session(sess,start_step)
            while not coord.should_stop():
                saver = tf.train.Saver(tf.global_variables())
                
                print(sess.run(tf.get_default_graph().get_tensor_by_name('L1/kernel:0')))
                print('Before restore loss = ', sess.run(simnet.loss))
                
                #restore_path = get_most_recent_checkpoint(log_dir)
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
            # 여기까지 왔으며, Error 메시지 보여도, 에러 아님.
            print('finally')
            coord.request_stop()        



###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
"""
SimpleNet의 init에서 datafeeder를 구조를 가지고 있다. train만 하는 것이 아니고, inference도 있기 때문
SimpleNet2로 개선

"""



class SimpleNet2():
    
    def __init__(self,hp,train_mode=False):
        self.train_mode=train_mode
        self.hp = hp

    def build_model(self, inputs,outputs=None):
        
        L1 = tf.layers.dense(inputs,units=hp.layer_size[0], activation = tf.sigmoid,name='L1')
        self.logits = tf.layers.dense(L1,units=hp.layer_size[1], activation = None,name='L2')
        self.preditions = tf.sigmoid(self.logits)
        if self.train_mode or outputs is not None:
            self.loss = tf.reduce_mean( 0.5*tf.square(self.preditions-outputs))
    
    def add_optimizer(self,global_step):
        # optimizer에 global_step을 넘겨줘야, global_step이 자동으로 증가된다.
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.hp.learning_rate).minimize(self.loss,global_step=global_step )
        return self.train_op

def run_and_save_SimpleNet2():
    # SimpleNet2 + DataFeeder   ----> train
    
    
    coord = tf.train.Coordinator()
    train_feeder = DataFeeder(coord,batch_size=2)
    simnet = SimpleNet2(hp,train_mode=True)  
    simnet.build_model(train_feeder.x, train_feeder.y)
    
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    simnet.add_optimizer(global_step)
    
    saver = tf.train.Saver(tf.global_variables())
    
    # Ctrl + C로 stop
    with tf.Session() as sess:
        try:
            
            sess.run(tf.global_variables_initializer())
            
            # 모델 restore
            if restore_path == '':
                start_step=0
                sess.run(tf.assign(global_step, 0))
            else:
                saver.restore(sess, restore_path)
                print('Resuming from checkpoint: %s' %(restore_path))
            
            start_step = sess.run(global_step)
            
            
            train_feeder.start_in_session(sess,start_step)  # 반드시 있어야됨
            while not coord.should_stop():
                step, _ =sess.run([global_step,simnet.train_op])
                 
                if step%1000==0:
                    print("step ",step, ": loss = ", sess.run(simnet.loss))
                
                if step%30000 ==0:
                    #print(tf.global_variables())
                    saver.save(sess, checkpoint_path, global_step=step)
                    print('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                    

        except Exception as e:
            print('Exiting due to exception: %s' % e)
            coord.request_stop(e)

        finally:
            print('finally')
            coord.request_stop()
            
            
def model_restore_SimpleNet2():
    # SimpleNet2 + DataFeeder   ----> train

    start_step = 0
    
    inputs = tf.placeholder(tf.float32, [None,3])
    simnet = SimpleNet2(hp,train_mode=False)  
    simnet.build_model(inputs)    
    
    with tf.Session() as sess:


        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        
        print(sess.run(tf.get_default_graph().get_tensor_by_name('L1/kernel:0')))
        print('prediction before training = ', sess.run(simnet.preditions,feed_dict={inputs: myDataX[:2] }))
        
        
    
        saver.restore(sess, restore_path)
        print('model restored!!!')
  
  
        print(sess.run(tf.get_default_graph().get_tensor_by_name('L1/kernel:0')))
        print('prediction after training = ', sess.run(simnet.preditions,feed_dict={inputs: myDataX[:2] }))
            
            

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
"""
DataFeeder class대신, tf.data.Dataset 을 이용하는 방식
SimpleNet2
위의 방식과 checkpoint 공유는 안된다. 별도로 선언한 global_step이 있어서..

"""
class DataFeeder2():
    def __init__(self,train_input,train_target, eval_input=None,eval_target=None,batch_size=128,buffer_size=50000,drop_remainder=False,num_epoch=3):
        self.train_input = train_input
        self.train_target = train_target
        self.eval_input = eval_input
        self.eval_target = eval_target
        
        self.buffer_size = buffer_size
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
    
    def mapping_fn(self,X, Y):
        inputs, labels = {'x': X}, Y
        return inputs, labels
    
    def train_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.train_input, self.train_target))
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size,drop_remainder=self.drop_remainder)
        dataset = dataset.repeat(count=self.num_epoch)
        dataset = dataset.map(self.mapping_fn)
        iterator = dataset.make_one_shot_iterator()
    
        return iterator.get_next()


    def eval_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.eval_input, self.eval_target))
        dataset = dataset.map(self.mapping_fn)
        dataset = dataset.batch(1,drop_remainder=self.drop_remainder)
        iterator = dataset.make_one_shot_iterator()
        
        return iterator.get_next()




def model_fn(features, labels, mode):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    batch_size = tf.shape(features['x'])[0]
    

    simplenet = SimpleNet2(hp,train_mode=TRAIN)                 
    simplenet.build_model(features['x'], labels)                 

    if PREDICT:
        predictions = {'predition': simplenet.predictions}
        
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    loss = simplenet.loss
    
    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)  # loss, eval_metric_ops른 넣었기 때문에 2개가 return
    
    if TRAIN:
        global_step = tf.train.get_global_step()
        train_op = simplenet.add_optimizer(global_step)
        logging_hook = tf.train.LoggingTensorHook({"global seep: ": global_step, "loss----" : loss }, every_n_iter=1000)
        return tf.estimator.EstimatorSpec(mode=mode,train_op=train_op,loss=loss,training_hooks = [logging_hook])

def run_and_save_SimpleNet3():
    # SimpleNet2 + DataFeeder2 + Estimator   ----> train
  
    # TensorFlow에서는 5가지의 로깅 타입을 제공하고 있습니다. ( DEBUG, INFO, WARN, ERROR, FATAL ) INFO가 설정되면, 그 이하는 다 출력된다.
    tf.logging.set_verbosity(tf.logging.INFO)   # 이게 있어야 train log가 출력된다.
    datfeeder = DataFeeder2(myDataX,myDataY,myDataX,myDataY,batch_size=2,num_epoch=2000)
    

    my_config =tf.estimator.RunConfig(log_step_count_steps=1000,save_summary_steps=10000,save_checkpoints_steps=3000)   # INFO:tensorflow:global_step/sec: 317.864  <--- 출력회수 제어
    est = tf.estimator.Estimator(model_fn=model_fn,model_dir=load_path,config = my_config) 



    print("="*10, "Train")
    est.train(datfeeder.train_input_fn)



    print("="*10, "Evaluation")
    eval_result = est.evaluate(input_fn = datfeeder.eval_input_fn,steps=2)  # steps는 최대 실행 횟수이다. data가 다 소진되면 steps를 다 채우지 못할 수도 있다.
    print('\nTest set loss: {loss:0.3f}\n'.format(**eval_result))
    
    print("="*10, "Test")
    predictions = est.predict(input_fn=datfeeder.eval_input_fn)   # user defined function이 아니면, lambda function으로 넘기면 안됨


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

# run_and_save_SimpleNet4()  ---> DataFeeder(Queue) + Model class  + tf.estimator
def run_and_save_SimpleNet4():
    # SimpleNet2 + DataFeeder + Estimator   ----> train
  
    # TensorFlow에서는 5가지의 로깅 타입을 제공하고 있습니다. ( DEBUG, INFO, WARN, ERROR, FATAL ) INFO가 설정되면, 그 이하는 다 출력된다.
    tf.logging.set_verbosity(tf.logging.INFO)   # 이게 있어야 train log가 출력된다.
    
    
    
    coord = tf.train.Coordinator()
    train_feeder = DataFeeder(coord,batch_size=2)
    sess = tf.Session()
    train_feeder.start_in_session(sess,0)
    
        
    def train_input_fn():
        x = tf.placeholder(tf.float32, shape=[None, 3], name='x')
        y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        return {'x': x}, y    # targets를 사용하려면...
    
    def feed_fn():
        batch_x, batch_y = sess.run([train_feeder.x,train_feeder.y])
        return {'x:0': batch_x, 'y:0': batch_y }
    
    
    my_config =tf.estimator.RunConfig(log_step_count_steps=1000,save_summary_steps=10000,save_checkpoints_steps=3000)   # INFO:tensorflow:global_step/sec: 317.864  <--- 출력회수 제어
    est = tf.estimator.Estimator(model_fn=model_fn,model_dir='hccho-ckpt\\model_ckpt',config = my_config) 
    

    print("="*10, "Train")
    est.train(train_input_fn,hooks=[tf.train.FeedFnHook(feed_fn)],steps=10000)




if __name__ == '__main__':
    #run_and_save_SimpleNet()    
    #model_restore_SimpleNet()
    
    ###########################
    ###########################
    #run_and_save_SimpleNet2()
    #model_restore_SimpleNet2()
    
    ###########################
    ###########################    


    
    run_and_save_SimpleNet3()
    #run_and_save_SimpleNet4()
    
    
    print("Done")
    
