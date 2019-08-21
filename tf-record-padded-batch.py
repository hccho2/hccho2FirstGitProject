# coding: utf-8
'''

'''
import os
import numpy as np
import tensorflow as tf
import pickle
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import librosa
from glob import glob
#Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space   ord('a')=97

voxforge_data_dir = './Voxforge'

#Some configs
num_features = 26
BATCH_SIZE = 32
if not os.path.isdir('./data'):
	os.makedirs('./data')



def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
   
def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def list_files_for_speaker(folder):
	'''
	Generates a list of wav files from the voxforge dataset.
	Args:
		###If want specific speaker
		speaker: substring contained in the speaker's folder name, e.g. 'Aaron'
		###
		folder: base folder containing the downloaded voxforge data

	Returns: list of paths to the wavfiles
	'''
	#If you want specific speaker, add speaker arg into function
	#speaker_folders = [d for d in os.listdir(folder) if speaker in d]
	speaker_folders = [d for d in os.listdir(folder)]
	wav_files = []

	for d in speaker_folders:
		for f in os.listdir(os.path.join(folder, d, 'wav')):
			wav_files.append(os.path.abspath(os.path.join(folder, d, 'wav', f)))

	return wav_files

def extract_features_and_targets(wav_file, txt_file):
	'''
	파일 1쌍 처리
	Extract MFCC features from an audio file and target character annotations from
	a corresponding text transcription
	Args:
		wav_file: audio wav file
		txt_file: text file with transcription

	Returns:
		features, targets, sequence length, original text transcription
	'''
	

	fs, audio = wav.read(wav_file)  # fs=16000, audio = array([-72, -86, -52, ..., -56, -63, -35], dtype=int16) --> 값의 범위가 e.g(-18066 ~ 13091)

	#y,sr = librosa.load(wav_file,sr=fs)   #lirosa는 -1~1 사이값
	
	features = mfcc(audio, samplerate=fs, numcep= num_features)  # hop_size = sr*0.01(default).   librosa로 뽑은 값을 넣어도 같은 결과(numerical 차이는 있음)

	#Tranform in 3D array
	features = np.asarray(features[np.newaxis, :])  # (624, num_features) --> (1, 624, num_features)
	features = (features - np.mean(features))/np.std(features)
	features_seq_len = features.shape[1]

	#Readings targets
	with open(txt_file, 'r') as f:
		for line in f.readlines():
			if line[0] == ';':
				continue

			#Get only the words between [a-z] and replace period for none
			original = ' '.join(line.strip().lower().split(' ')).replace('.', '').replace("'", '').replace('-', '').replace(',','')
			targets = original.replace(' ', '  ')
			targets = targets.split(' ')
	   
	#Adding blank label
	targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])  # ['once', '', 'there', '', 'was', '', 'a', '', 'young', '', 'rat', ''...]  --> array(['o', 'n', 'c', 'e', '<space>', 't', 'h',  ...])

	#Transform char into index
	targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])  # ascii 코드를 이용하여, index로 만듬. array([15, 14,  3,  5,  0, 20,  8,  5, 18,  5,  0...]
	return features, targets, features_seq_len, original



def make_tfrecods(wav_files,tfrecord_filename):
	batched_data = []
	original_targets = []
	
	np.random.shuffle(wav_files)
	
	batch_features = []
	batch_targets = []
	batch_seq_len = []
	batch_original = []
	
	for f in wav_files:

		txt_file = f.replace('\\wav\\', '\\txt\\').replace('.wav', '.txt')
		features, targets, seq_len, original = extract_features_and_targets(f, txt_file)

		batch_features.append(features[0].reshape(-1).tolist())  # tfrecord 파일은 1-dim만 수용. np.array는 안됨 list로 변환 필요
		batch_targets.append(targets.tolist())
		batch_seq_len.append(seq_len)
		batch_original.append(original)  # string
	

	writer = tf.python_io.TFRecordWriter(tfrecord_filename)
	for i in range(len(batch_features)):
		example = tf.train.Example(features=tf.train.Features(feature={
			'features': _float_feature(batch_features[i]),
			'targets':_int64_feature(batch_targets[i]),
			'seq_len':_int64_feature(batch_seq_len[i]),
			'original':_bytes_feature(tf.compat.as_bytes(batch_original[i]))
		}))
		writer.write(example.SerializeToString())
	writer.close()
	

def read_tfrecords(filenames):
	batch_size=2
	if not isinstance(filenames, list):
		filenames = [filenames]
	dataset = tf.data.TFRecordDataset(filenames)  # 1개의 파일이나 여러개의 파일이 들어갈 수 있다.
	def _parse_function(example_proto):
	# tf.VarLenFeature: Configuration for parsing a variable-length input feature.
	# 참고로 FixedLenFeature도 있다.

		keys_to_features = {'features':tf.VarLenFeature(tf.float32),'targets':tf.VarLenFeature(tf.int64),
						   'seq_len':tf.VarLenFeature(tf.int64),'original':tf.FixedLenFeature([], tf.string)}
		parsed_features = tf.parse_single_example(example_proto, keys_to_features)
		
		# 여기서는 자유롭게 return하지만, tf.Estimator의 train, evaluation, predict에 넘기기 위해서는 featrues(dict), lables 2개로 묶어야 한다.
		return tf.sparse.to_dense(parsed_features['features']), tf.sparse.to_dense(parsed_features['targets']), tf.sparse.to_dense(parsed_features['seq_len']) , parsed_features['original']
	# Parse the record into tensors.
	dataset = dataset.map(_parse_function)
	# Shuffle the dataset
	dataset = dataset.shuffle(buffer_size = 50)  # buffer_size만큼 만들어 놓고, 그 중에서 random하게 뽑아온다. 뽑아진 것을 대신하여 하나를 더 채운다.
	# Repeat the input indefinitly
	dataset = dataset.repeat()  
	dataset = dataset.prefetch(buffer_size = 20)  # data를 미리 만들어 놓는 갯수

	# Generate batches
	# padded_shapes: None이면 가장 큰 data 기준.
	dataset = dataset.padded_batch(batch_size, padded_shapes=([None],[None],[None],[]), padding_values=(tf.constant(-1.1, dtype=tf.float32),tf.constant(-99, dtype=tf.int64),tf.constant(0, dtype=tf.int64),tf.constant("", dtype=tf.string)))
	#dataset = dataset.batch(2)
	# Create a one-shot iterator
	iterator = dataset.make_one_shot_iterator()
	i,j,k,l = iterator.get_next()

	with tf.Session() as sess:
		
		ii,jj,kk,ll = sess.run([i,j,k,l])
		ii = ii.reshape(batch_size,-1,num_features)
		print(ii.shape,jj,kk,ll)
		
		
		
		ii,jj,kk,ll = sess.run([i,j,k,l])
		ii = ii.reshape(batch_size,-1,num_features)
		print(ii.shape,jj,kk,ll)
		
def read_npz(filenames):
	'''
		['audio', 'mel', 'linear', 'time_steps', 'mel_frames', 'text', 'tokens', 'loss_coeff', 'allow_pickle']
		X['time_steps'],  (66600)
		X['mel'].shape, (222, 80)
		X['audio'].shape, (66600)
		X['linear'].shape, (222, 1025)
		X['mel_frames']     222
		X['tokens'].shape (39,)
	
	'''
	
	batch_size=2

	if not isinstance(filenames, list):
		filenames = [filenames]
	
	dataset = tf.data.Dataset.from_tensor_slices(filenames)  # npz파일은 tf.data.TFRecordDataset를 사용할 수 없다.
	
	def _parse_function(filename):
		# 파일을 읽어들이는 작업이기 때문에, map을 먼저 설정하고, batch(2)가 되어야 한다.
		# padded_batch로 넘어 가려면, 1-dim array로 return 되어야 한다.
		# 길이가 다르기 때문에, padded_batch가 아닌, batch(2)로 넘어가면 안 된다.
		def get_data_from_npz(filename_):
			# padded_batch에 들어가야 되므로, 1-dim array로 바뀌어야 한다.
			data = np.load(filename_)
			return data['audio'],data['mel'].reshape(-1) # ,data['tokens'],data['text']	  <------ 아래 read_npz2에 보면, 꼭 1차원으로 변형하지 않아도 된다.
		
		audio, mel = tf.py_func(get_data_from_npz, [filename], (tf.float32,tf.float32))
		return audio,mel
	
	

	dataset = dataset.map(_parse_function)		
	dataset = dataset.shuffle(buffer_size = 50)  # buffer_size만큼 만들어 놓고, 그 중에서 random하게 뽑아온다. 뽑아진 것을 대신하여 하나를 더 채운다.
	# Repeat the input indefinitly
	dataset = dataset.repeat()  
	dataset = dataset.prefetch(buffer_size = 20)  # data를 미리 만들어 놓는 갯수

	# Generate batches
	# padded_shapes: None이면 가장 큰 data 기준.
	
	# batch로 묶인다는 것은 padding이 이루어져야 하는 array거나, padding이 의미없는 single number들이다.
	dataset = dataset.padded_batch(batch_size, padded_shapes=([None],[None]), padding_values=(tf.constant(-1.1, dtype=tf.float32),tf.constant(-2.2, dtype=tf.float32)))
	# Create a one-shot iterator
	iterator = dataset.make_one_shot_iterator()
	i,j = iterator.get_next()
	
	
	with tf.Session() as sess:
		
		ii,jj = sess.run([i,j])
		print(ii.shape,jj.shape)

		ii,jj = sess.run([i,j])
		print(ii.shape,jj.shape)

def gen_test():
	
	def gen():
		for i in range(5):
			if i%2==0:
				continue
			yield(i)
	
	dataset = tf.data.Dataset.from_generator(gen,output_types=(tf.int32),output_shapes=tf.TensorShape([]))
	dataset = dataset.batch(2,drop_remainder=True)
	dataset = dataset.repeat(100)   # 여기서 repeat를  하든지, 아니면, gen()함수의 for loop를 while True:로 돌리든지 해야 한다.  repeat를 할 경우, repeate 회수가 끝나면 Error발생함.
	
	
	iterator = dataset.make_one_shot_iterator()
	x= iterator.get_next()
	
	sess = tf.Session()
	
	for i in range(10):
		print(sess.run(x))
	
def read_npz2(filenames):
	# tf.data.Dataset.from_generator()이용
	# tfrecord 파일은 1차원으로 변형해야 하지만, npz는 형태를 보존하여 처리할 수 있다.
	def _gen_data():
		for f in filenames:
			data = np.load(f)
			#print(f,data['audio'].shape)
			yield({"audio_x": data['audio'],"mel_x":data['mel'],"length_x": data['mel_frames']})
			#yield({"audio_x": data['audio'],"mel_x":data['mel'],"length_x": [data['mel_frames']]})  # length_x 를 list로 묶으면 [] 아래의 tf.TensorShape([1])이 되어야 한다.
			
	
	
	output_types=({"audio_x": tf.float32, "mel_x": tf.float32,"length_x": tf.int32})
	
	# length_x에서는 tf.TensorShape([]) 이어야 한다. tf.TensorShape([None])이나 tf.TensorShape([1])은 안된다.
	output_shapes=({"audio_x": tf.TensorShape([None]), "mel_x": tf.TensorShape([None,80]), "length_x": tf.TensorShape([])})
	dataset = tf.data.Dataset.from_generator(_gen_data,output_types=output_types,output_shapes=output_shapes)
	
	
	dataset = dataset.repeat(20)
	#dataset = dataset.batch(1)
	
	# batch를 만들기 전의 shape이다.
	padded_shapes=({"audio_x": tf.TensorShape([None]), "mel_x": tf.TensorShape([None, 80]), "length_x": tf.TensorShape([])})
	dataset = dataset.padded_batch(2,padded_shapes=padded_shapes)
	
	dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

	iterator = dataset.make_one_shot_iterator()
	i= iterator.get_next()


	with tf.Session() as sess:
		ii = sess.run(i)
		print(ii['audio_x'].shape, ii['mel_x'].shape, ii['length_x'])

		ii = sess.run(i)
		print(ii['audio_x'].shape, ii['mel_x'].shape, ii['length_x'])

		ii = sess.run(i)
		print(ii['audio_x'].shape, ii['mel_x'].shape, ii['length_x'])

if __name__ == '__main__':
	
	tfrecord_filename = './data/train_data_batched1.tfrecords'
	tfrecord_filenames = ['./data/train_data_batched1.tfrecords','./data/train_data_batched2.tfrecords']
	
	#wav_files = list_files_for_speaker(voxforge_data_dir)  # 디렉토리에 상관없이 wav파일들을 list로 만듬  ['D:\\hccho\\Tensorflow-Speech-to-Text\\Voxforge\\1028-20100710-hne\\wav\\ar-01.wav', 'D:\\hccho\\Tensorflow-Speech-to-Text\\Voxforge\\1028-20100710-hne\\wav\\ar-02.wav', ...]
	#make_tfrecods(wav_files,tfrecord_filename)
	
	
	
	#read_tfrecords(tfrecord_filename)
	#read_tfrecords(tfrecord_filenames)
	
	
	
	npz_filenames = glob("{}/*.npz".format('.\\data'))
	#print(npz_filenames)
	#read_npz(npz_filenames)
 	
 
	#read_npz2(npz_filenames)
	
	
	gen_test()
	
	print('Done')
	
	
