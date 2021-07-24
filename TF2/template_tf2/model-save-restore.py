
'''
hp['slack_send_msg'] = True/False로 slack으로 message보낼지 결정.

> python model-save-restore.py --test
> python model-save-restore.py --train --load_path hccho-ckpt\hccho-mm-2020-08-24_10-50-03
> python model-save-restore.py --load_path hccho-ckpt\hccho-mm-2020-08-24_10-50-03


'''



import argparse
import infolog
from hyper_params import hp
from utils import prepare_dirs,prepare
import tensorflow as tf
import os





def train():
    load_path = args.load_path
    log, load_path,restore_path,checkpoint_path = prepare(hp,load_path)
    print = log  # slack으로 message가 간다.
    print(load_path+'\t'+restore_path+'\t'+checkpoint_path)
    
    
    
    
    # model build
    input_dim = 3
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=10,input_dim=input_dim,activation='relu'))
    model.add(tf.keras.layers.Dense(units=1,activation=None))
    
    
    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer,loss='mse')
    
    
    
    # checkpoint setting
    checkpoint = tf.train.Checkpoint(model=model,optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, load_path,checkpoint_name=hp['ckpt_file_name_preface'] ,max_to_keep=5)
    
    if ckpt_manager.latest_checkpoint:
        # model이 build되지 않았지만, build되면  restore된다.  ---> Delayed restorations
        checkpoint.restore(ckpt_manager.latest_checkpoint)#.expect_partial()
        step_count = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        #start_epoch = step_count // (len(train_input)//batch_size) + 1
        start_epoch = step_count
        print ('Latest checkpoint restored!! {},  epochs = {}, step_count = {}'.format(ckpt_manager.latest_checkpoint, start_epoch, step_count))
        start_epoch +=1
    else:
        step_count = 0
        start_epoch = 1
        print("Initializing from scratch.")
    
    
    X = tf.random.normal(shape=(10, input_dim))
    Y = tf.random.normal(shape=(10, 1))
    
    for epoch in range(start_epoch, start_epoch+hp['num_epoch']):
        
        history  = model.fit(X,Y,batch_size=hp['batch_size'], epochs=1,verbose=1,validation_split=0.1)
        print('epoch: {}/{}, loss: {}, val_loss: {}'.format(epoch,start_epoch+hp['num_epoch']-1,history.history['loss'],history.history['val_loss']),slack=hp['slack_send_msg'])
        step_count += 1
    
        if (epoch)%5==0:
            ckpt_save_path = ckpt_manager.save(checkpoint_number = epoch)   # epoch으로 할지? step_count로 할지...
            print ('Saving checkpoint for epoch {} at {}'.format(epoch,ckpt_save_path),slack=hp['slack_send_msg'])


def test():
    print('test' * 10)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    #parser.add_argument('--load_path', default=None)
    parser.add_argument('--load_path', default='hccho-ckpt\\hccho-mm-2020-08-24_10-50-03')

    args = parser.parse_args()
    print(args)
    
    
    

    
    
    if args.train_flag:
        train()
    else:
        test()
    
    
