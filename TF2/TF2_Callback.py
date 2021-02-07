
'''


'''



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time
# https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_metrics   ---> 사용자 정의 metric 만들기.
def mymetric(y_true, y_pred):  # y_true, y_pred
    return tf.reduce_mean(tf.square(y_true-y_pred))
mymetric.__name__ = 'hccho_metric'   # 지정하지 않으면, 그냥 함수 이름.


class MyCallback(tf.keras.callbacks.Callback):
        
    def on_train_begin(self, logs={}):
        # This method will be called when the training start.
        # Therefore, we use it to initialize some elements for our Callback:
        self.logs = dict()  # {'loss': [3.96, 1.327, ...], 'hccho_metric': [3.9433, 1.331, ...], 'val_loss': [1.396, 1.058, ...], 'val_hccho_metric': [1.392, 1.06, ...]}
        self.fig, self.ax = None, None
 
    def on_epoch_end(self, epoch, logs={}): # logs = {'loss': 1.3536112308502197, 'hccho_metric': 1.3536112308502197}
        # This method will be called after each epoch.
        # Keras will call this function, providing the current epoch number,
        # and the values of the various losses/metrics for this epoch (`logs` dict).
        
        # We add the new log values to the list...
        for key, val in logs.items():
            if key not in self.logs:
                self.logs[key] = []
            self.logs[key].append(val)
        # ... then we plot everything:
        self._plot_logs()
 
    def on_train_end(self, logs={}):
        pass    # our callback does nothing special at the end of the training
 
    def on_epoch_begin(self, epoch, logs={}):
        pass   # ... nor at the beginning of a new epoch
 
    def on_batch_begin(self, batch, logs={}):
        pass   # ... nor at the beginning of a new batch
 
    def on_batch_end(self, batch, logs={}):
        pass   # ... nor after.
    
    def _plot_logs(self):
        # Method to clear the figures and draw them over with new values:
        if self.fig is None: # First call - we initialize the figure:
            num_metrics = len(self.logs) 
            self.fig, self.ax = plt.subplots(math.ceil(num_metrics / 2), 2, figsize=(10, 8))
            self.fig.show()
            self.fig.canvas.draw()
        
        # Plotting:
        i = 0
        for key, val in self.logs.items():
            id_vert, id_hori = i // 2, i % 2
            self.ax[id_vert, id_hori].clear()
            self.ax[id_vert, id_hori].set_title(key)
            self.ax[id_vert, id_hori].plot(val)
            i += 1
        
        #self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.75, bottom=0.25)
        self.fig.canvas.draw()
        plt.pause(1e-17)



def plot_callback():
    batch_size = 16
    input_dim = 300
    
    inputs = tf.keras.Input(shape=(input_dim,))  # 구제적인 입력 data없이 ---> placeholder같은 ...
    
    L1 = tf.keras.layers.Dense(units=500,input_dim=3,activation='relu')
    L2 = tf.keras.layers.Dense(units=1,activation=None)
    
    outputs = L2(L1(inputs))
    
    model = tf.keras.Model(inputs = inputs,outputs = outputs)  # model.input, model.output 
    print(model.summary())
    
    
    X = tf.random.normal(shape=(1000, input_dim))
    Y = tf.random.normal(shape=(1000, 1))
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer,loss='mse',metrics=[mymetric])
    
    history = model.fit(X,Y,epochs=50,verbose=1,batch_size=batch_size, validation_data=(X,Y), callbacks=[MyCallback()])



class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.s_time = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        print(f"\t*****elapsed: {time.time()-self.s_time:.2f}")
def time_callback():
    batch_size = 16
    input_dim = 300
    
    inputs = tf.keras.Input(shape=(input_dim,))  # 구제적인 입력 data없이 ---> placeholder같은 ...
    
    L1 = tf.keras.layers.Dense(units=500,input_dim=3,activation='relu')
    L2 = tf.keras.layers.Dense(units=1,activation=None)
    
    outputs = L2(L1(inputs))
    
    model = tf.keras.Model(inputs = inputs,outputs = outputs)  # model.input, model.output 
    
    X = tf.random.normal(shape=(1000, input_dim))
    Y = tf.random.normal(shape=(1000, 1))
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer,loss='mse',metrics=[mymetric])
    time_callback = TimeHistory()
    history = model.fit(X,Y,epochs=20,verbose=1,batch_size=batch_size, validation_data=(X[:20],Y[:20]), callbacks=[time_callback])
    print(time_callback.times)


if __name__ == "__main__":
    time_callback()




