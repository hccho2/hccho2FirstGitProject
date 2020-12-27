
'''
https://www.tensorflow.org/guide/data   ---> 일반적인 사용법.

https://www.tensorflow.org/guide/data_performance  ----> 대용량을 다루기 위해서...


다음은 성능이 좋은 텐서플로 입력 파이프라인을 설계하기 위한 가장 좋은 예제를 요약한 것입니다:

1. prefetch 변환을 사용하여 프로듀서와 컨슈머의 작업을 오버랩하세요.
2. interleave 변환을 이용해 데이터 읽기 변환을 병렬화하세요.
3. num_parallel_calls 매개변수를 설정하여 map 변환을 병렬 처리하세요.
4. 데이터가 메모리에 저장될 수 있는 경우, cache 변환을 사용하여 첫 번째 에포크동안 데이터를 메모리에 캐시하세요.
5. map 변환에 전달된 사용자 정의 함수를 벡터화하세요.
5. interleave, prefetch, 그리고 shuffle 변환을 적용하여 메모리 사용을 줄이세요.



'''



import tensorflow as tf
import time
def data_performance_test():
    class ArtificialDataset(tf.data.Dataset):
        def _generator(num_samples):
            # 파일 열기  ----> 여기서 파일을 1개씩 여는 역할
            time.sleep(0.03)
            
            for sample_idx in range(num_samples):
                # 파일에서 데이터(줄, 기록) 읽기
                time.sleep(0.015)
                
                yield (sample_idx,)
        
        def __new__(cls, num_samples=3):
            # 왜 __init__으로 안 했을까???   __new__는 __init__보다 먼저 call되고, object마다 생성되는 것은 아니다. 딱 1번만 call된다.
            return tf.data.Dataset.from_generator(
                cls._generator,
                output_types=tf.dtypes.int64,
                output_shapes=(1,),
                args=(num_samples,)
            )
    
    
    dataset = ArtificialDataset(4)
    
    
    start_time = time.perf_counter()
    for i,d in enumerate(dataset):
        print(i,d)
    tf.print("실행 시간1:", time.perf_counter() - start_time)
    
    
    def benchmark(dataset, num_epochs=2):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            for sample in dataset:
                # 훈련 스텝마다 실행
                time.sleep(0.01)
        tf.print("실행 시간==:", time.perf_counter() - start_time)
    
    
    
    benchmark(ArtificialDataset())
    
    
    benchmark(ArtificialDataset().prefetch(tf.data.experimental.AUTOTUNE))
    
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    start_time = time.perf_counter()
    for i,d in enumerate(dataset):
        print(i,d)
    tf.print("실행 시간:", time.perf_counter() - start_time)



def mnist_dataset_test():
    # mnist 데이터 가져오기 및 포맷 맞추기
    # C:\Users\BRAIN\.keras\datasets  ---> mnist.npz(11M) 다운받음
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # numpy array 
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
    
    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(x_train[...,tf.newaxis]/255, tf.float32), tf.cast(y_train,tf.int64)))
    dataset = dataset.shuffle(1000).batch(32)
    
    
    
def squence_test():
    # 대용량 Dataset 다루기
    # tf.keras.utils.Sequence  ---> pytorch의 Dataset + DataLoader 과 유사
    
    pass  # TF2_RNN.py에 example 있음.


if __name__ == '__main__':
    #data_performance_test()
    
    mnist_dataset_test()




