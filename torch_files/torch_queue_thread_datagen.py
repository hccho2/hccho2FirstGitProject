# coding: utf-8
'''


'''

import torch
import numpy as np
import threading,queue
from torch.multiprocessing import Process, Queue, Pool
import traceback


def prefetch_data(queue, batch_size,num_features):
    print("start prefetching data...")
    while True:
        try:
            data = torch.from_numpy(np.random.rand(batch_size,num_features))
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def init_jobs(queue, batch_size,num_features):
    # Process를 여러개 만들수도 있다. Peocess list로
    task = Process(target=prefetch_data, args=(queue, batch_size,num_features))

    task.daemon = True
    task.start()
    return task



def data_preprocessing(data_queue, processed_training_queue, sema):
    while True:
        data = data_queue.get()
        
        # 필요한 조작을 한다. (e.g. tensor를 pin momory로 올린다.)
        data = data*1000
        processed_training_queue.put(data)
    
        if sema.acquire(blocking=False):
            return

def train():
    batch_size=2
    num_features=3
    
    training_queue   = Queue(3)
    training_tasks = init_jobs(training_queue, batch_size,num_features)
    
    processed_training_queue = queue.Queue(3)
    
    training_semaphore   = threading.Semaphore()
    training_semaphore.acquire()
    
    
    training_thread = threading.Thread(target=data_preprocessing, args=(training_queue,processed_training_queue,training_semaphore))
    training_thread.daemon = True
    training_thread.start()
    

    
    
    for i in range(4):
        data = processed_training_queue.get(block=True)
        print(data)
    
    
    
    training_tasks.terminate()

if __name__ == '__main__':
    train()