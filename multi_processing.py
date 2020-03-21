import numpy as np
import random
from collections import deque
import multiprocessing
import multiprocessing.connection
class MyGame:
    # 실질적인 multi processing에서 일을 하는 process
    def __init__(self,seed):
        self.seed = seed
    
    def gen_data(self,mu,sigma):
        return sigma * np.random.randn(2,3) + mu
    

class Worker(object):

    child: multiprocessing.connection.Connection
    process: multiprocessing.Process
    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()
        
def worker_process(remote: multiprocessing.connection.Connection, seed: int):                                               
    game = MyGame(seed)

    while True:
        cmd, data = remote.recv()
        if cmd == "gen_dataX":
            remote.send(game.gen_data(*data))  # MyGame.gen_data

        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


def main():
    
    n_workers = 4
    
    workers = [Worker(i) for i in range(n_workers)]
    
    data = np.zeros((n_workers, 2,3), dtype=np.float32)
    
    for _ in range(5):
        
        # 아래 2번의 for loop로 분산처리로 data를 생성한다.
        for i,worker in enumerate(workers):
            worker.child.send(("gen_dataX", (i*10,1.0)))
        
        for i,worker in enumerate(workers):
            data[i] = worker.child.recv()
    
    
        # 생성된 data로 일을 처리하면 된다.
        print(data.shape)
        print(data)
    
    
    # process closing
    for worker in workers:
        worker.child.send(("close", None))
    
    
    

if __name__ == "__main__":
    main()
