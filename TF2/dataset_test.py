
'''
https://www.tensorflow.org/guide/data   ---> 일반적인 사용법.

https://www.tensorflow.org/guide/data_performance  ----> 대용량을 다루기 위해서...



https://www.tensorflow.org/guide/data_performance?hl=ko#%EA%B0%80%EC%9E%A5_%EC%A2%8B%EC%9D%80_%EC%98%88%EC%A0%9C_%EC%9A%94%EC%95%BD
다음은 성능이 좋은 텐서플로 입력 파이프라인을 설계하기 위한 가장 좋은 예제를 요약한 것입니다:

1. prefetch 변환을 사용하여 프로듀서와 컨슈머의 작업을 오버랩하세요.
2. interleave 변환을 이용해 데이터 읽기 변환을 병렬화하세요.        tf.data.Dataset.interleave
3. num_parallel_calls 매개변수를 설정하여 map 변환을 병렬 처리하세요. datasets.map(map_fn, num_parallel_calls=3)
4. 데이터가 메모리에 저장될 수 있는 경우, cache 변환을 사용하여 첫 번째 에포크동안 데이터를 메모리에 캐시하세요.
5. map 변환에 전달된 사용자 정의 함수를 벡터화하세요.
5. interleave, prefetch, 그리고 shuffle 변환을 적용하여 메모리 사용을 줄이세요.



** 한번에 memory에 올릴 수 없는 data의 반복적인 loading은 generator보다는 map에서 처리하는 것이 효율적인다. generator에서 file이름을 생성하고,map에서 loading한다. 


'''


import numpy as np
import tensorflow as tf
import time
import glob
import librosa
import random


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




def map_fn_test():
    # 결론: mapping function을 randomness는 epoch마다 달라진다.
    X = np.random.randn(4,2)
    Y = np.arange(4)
    
    def map_fn(X,Y):
        return (X + tf.random.normal(X.shape,dtype=X.dtype)*0.01,Y)
    
    
    dataset = tf.data.Dataset.from_tensor_slices((X,Y))
    
    dataset = dataset.map(map_fn).repeat(2)
    dataset = dataset.batch(2)
    
    
    for d in dataset:
        print(d)
    

    print('=='*10)
    print(X)



def image_model_test():
    from tensorflow.keras.applications import resnet
    X = np.random.randint(0,255,size=(4,224,224,3))
    Y = np.arange(4)
    dataset = tf.data.Dataset.from_tensor_slices((X,Y))
    
    
    
    model = resnet.ResNet50(weights=None,include_top=False,input_shape=(224,224,3))
    
    out = resnet.preprocess_input(X)
    
    
    #print(X)
    #print(out)
    
    dataset = dataset.map(lambda a,b: (resnet.preprocess_input(a), b))
    dataset = dataset.batch(2)
    
    
    for x,y in dataset.take(2):
        print(x.shape,y.shape)
    
def padded_batch_test():

    # Components of nested elements can be padded independently.
    elements = [([1, 2, 3, 6], [10]),
                ([4, 5], [11, 12])]
    dataset = tf.data.Dataset.from_generator(lambda: iter(elements), (tf.int32, tf.int32))
    # Pad the first component of the tuple to length 4, and the second
    # component to the smallest size that fits.
    dataset = dataset.padded_batch(2,padded_shapes=([4], [None]),padding_values=(-1, 100))
    
    A = list(dataset.as_numpy_iterator())
    
    print(A)




def generator_test():
    
    wav_files = glob.glob(r'D:\hccho\CommonDataset\kss_small\kss\1\*.wav')

    print(wav_files)
    
    wav_files = ['aa.wav', 'xxx.wav']
    #wav_files = [3.5,2.4]

    def gen(files):
        for i, f in enumerate(files):
            yield f

    def gen2(files):
        for i, f in enumerate(files):
            yield {"a": i, "b": f}  # output_types에서 지정한 dict key와 일치해야 한다.

    def gen3(files):
        for i, f in enumerate(files):
            yield [i]*(i+3),f  

            
    # output_types: generator가 return하는 type
    datasets = tf.data.Dataset.from_generator(gen,output_types=tf.string, args=([wav_files])) # args는 []로 감싼 후, 넘겨야 한다.
    datasets2 = tf.data.Dataset.from_generator(gen2,output_types={"a": tf.int32, "b": tf.string}, args=([wav_files]))
    datasets3 = tf.data.Dataset.from_generator(gen3,output_types=(tf.int32, tf.string), args=([wav_files]))
    
    def map_fn(wav_files):
        
        return wav_files
    
    for x in datasets.take(2):
        print(x)

    print('='*20)
    for x in datasets2.take(2):
        print(x)

    print('='*20)
    for x in datasets3.take(2):
        print(x)


def generator_test2():
    wav_files = glob.glob(r'D:\hccho\CommonDataset\kss_small\kss\1\*.wav')

    def gen(files):
        # python 함수 사용 가능
        for i, f in enumerate(files):
            yield f
  
    def map_fn(wav_file):
        # python 함수를 사용하기 위해서는 tf.numpy_function을 사용해야 된다.
        # 이 경우는 test 목적으로 만든 것. 효율적인 방식은 아님. 매번 반복해서, wav 파일을 librosa가 load하여, mel_spectrogram으로 변환할 필요가 없다.
        # 한번 load해서 npy로 저장해 놓응 것을, 여기서는 불러와서 return하는 방식을 사용해야 한다.
        scale, sr = tf.numpy_function(librosa.load,[wav_file],(tf.float32,tf.int32))
        mel_spectrogram = tf.numpy_function(lambda x,y: librosa.feature.melspectrogram(x, sr=y, n_fft=2048, hop_length=512, n_mels=10), [scale,sr],tf.float32)  # n_mels,n_frames
        return tf.transpose(mel_spectrogram)
            
    # output_types: generator가 return하는 type
    datasets = tf.data.Dataset.from_generator(gen,output_types=tf.string, args=([wav_files])) # args는 []로 감싼 후, 넘겨야 한다.

    datasets = datasets.map(map_fn)

    
    for x in datasets.take(2):
        print(x.shape)


def generator_test3():
    data_files = glob.glob(r'D:\hccho\CommonDataset\kss_small\dump_kss\train\ids\*.npy')

    def gen(data_files):
        # padding을 최소화하기 위해, 길이로 정렬 후, feeding
        random.shuffle(data_files)
        buffer_size = 8  # 대략적인 batch size보다 충분히 크게...
        
        n_iter = len(data_files) //(buffer_size)
        for i in range(n_iter):
            examples =  [np.load(x) for x in data_files[i*buffer_size:(i+1)*buffer_size] ]
            examples.sort(key=lambda x: len(x))
        
            for e in examples:
                yield e
  

            
    # output_types: generator가 return하는 type
    datasets = tf.data.Dataset.from_generator(gen,output_types=tf.int32, args=([data_files])) # args는 []로 감싼 후, 넘겨야 한다.

    datasets = datasets.padded_batch(2,padded_shapes=[None],padding_values=0)
    datasets = datasets.repeat()

    
    for x in datasets.take(20):
        print(x.shape, x)
        print('-'*5)




if __name__ == '__main__':
    data_performance_test()
    
    #mnist_dataset_test()

    #map_fn_test()
    #image_model_test()
    
    #padded_batch_test()
    #generator_test()
    #generator_test2()
    #generator_test3()

