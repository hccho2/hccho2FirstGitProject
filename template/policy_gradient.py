'''


'''
import numpy as np
import tensorflow as tf
import gym
import os,time
import matplotlib.pyplot as plt
from collections import deque
env = gym.make("Pong-v0")

gamma = .99

CHECK_POINT_DIR = './pong'
CHECK_POINT_PREFACE = 'pong'

# Constants defining our neural network
input_size = 80 * 80 * 4
action_space = env.action_space.n
print("Pong Action space", action_space)
global_step_op = tf.train.get_or_create_global_step()
with tf.name_scope("cnn"):
    X = tf.placeholder(tf.float32, [None, input_size], name="input_x")
    #x_image = tf.reshape(X, [-1, 80, 80, 4])
    x_image = tf.transpose(tf.reshape(X, [-1, 4, 80, 80]),[0,2,3,1])
    
    
    tf.summary.image('input', x_image, 8)

    init = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
    #init = tf.contrib.layers.xavier_initializer()  # relu -----> xavier 사용하면 안됨.

    # Build a convolutional layer random initialization
    W_conv1 = tf.get_variable("W_conv1", shape = [5, 5, 4, 32], initializer=init) 
    # W is [row, col, channel, feature]
    b_conv1 = tf.Variable(tf.zeros([32]), name="b_conv1")
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='VALID') + b_conv1, name="h_conv1")

    W_conv2 = tf.get_variable("W_conv2", shape = [5, 5, 32, 64], initializer=init)
    b_conv2 = tf.Variable(tf.zeros([64]), name="b_conv2")
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")

    W_conv3 = tf.get_variable("W_conv3", shape = [5, 5, 64, 64], initializer=init)
    b_conv3 = tf.Variable(tf.zeros([64]), name="b_conv3")
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 2, 2, 1], padding='VALID') + b_conv3, name="h_conv3")
    
    # Build a fully connected layer with softmax 
    h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*64], name="h_pool2_flat")
    W_fc1 = tf.get_variable("W_fc1", shape = [7*7*64, action_space], initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.Variable(tf.zeros([action_space]), name = 'b_fc1')
    action_pred = tf.nn.softmax(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")

    tf.summary.histogram("action_pred", action_pred)

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(tf.float32, [None, action_space], name="input_y")
advantages = tf.placeholder(tf.float32, [None, 1], name="reward_signal")

# Loss function
# Sum (Ai*logp(yi|xi))
log_lik = -Y * (tf.log(tf.clip_by_value(action_pred, 1e-10 , 1.0)))
loss = tf.reduce_mean(tf.reduce_sum(log_lik * advantages, axis=1))


# Learning
train = tf.train.AdamOptimizer().minimize(loss,global_step=global_step_op)




def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward
        http://karpathy.github.io/2016/05/31/rl/  """
    discounted_r = np.zeros_like(r)  #(84,1)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # compute the discounted reward backwards through time
    # standardize the rewards to be unit normal (helps control the gradient
    # estimator variance)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
        http://karpathy.github.io/2016/05/31/rl/ """
        
    # background color (109,118,43) 인데, 첫번째 channel만 살린 후 제거하기 때문에, 109만 날리면 된다.
    I = I[35:195]  # crop  위 아래 불필요한 부눈 잘라내기     (210, 160, 3) --> (160, 160, 3)
    I = I[::2, ::2, 0]  # downsample by factor of 2  ---> channel 1개만 살린다.
    I[I == 144] = 0  # erase background (background type 1)   env.step()이 return하는 state의 background는  (144,72,17)
    I[I == 109] = 0  # erase background (background type 2)   env.reset()이 return하는 state의 background는  (109,118,43)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


# Setting up our environment

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Savor and Restore
saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint(CHECK_POINT_DIR)  # tf.train.get_checkpoint_state(CHECK_POINT_DIR)
if checkpoint:
    try:
        saver.restore(sess, checkpoint)
        print("Successfully loaded:", checkpoint)
    except:
        print("Error on loading old network weights")
    global_step = sess.run(global_step_op)
else:
    print("Could not find old network weights")
    global_step = 0
s_time = time.time()
reward_sum_history = deque(maxlen = 100)
while True:
    global_step += 1

    xs_list = []
    ys_list = []
    rewards = np.empty(0).reshape(0, 1)
    ep_rewards_list = []

    reward_sum = 0
    state = env.reset()
    state = prepro(state)  # (210, 160, 3) --> (6400,)

    # Initial 4 frame data
    s_t = np.array([state, state, state, state])

    while True:
        #env.render()
        # Append the observations to our batch
        x = np.reshape(s_t, [1, input_size])

        # Run the neural net to determine output
        action_prob = sess.run(action_pred, feed_dict={X: x})
        action_prob = np.squeeze(action_prob)  # shape (?, n) -> n
        action = np.random.choice(action_space, size=1, p=action_prob)[0]
        
        #random_noise = np.random.uniform(0, 1, output_size)
        #action = np.argmax(action_prob + random_noise)
        # print("Action prediction: ", np.argmax(action_prob), " action taken:", action,
        #      np.argmax(action_prob) == action)

        # Append the observations and outputs for learning
        xs_list.append(x)
        y = np.eye(action_space)[action:action + 1]  # One hot encoding
        ys_list.append(y)

        state, reward, done, _ = env.step(action)
        state = prepro(state)
        s_t = np.array([state, s_t[0], s_t[1], s_t[2]])  # s_t[4] out!
        reward_sum += reward

        ep_rewards_list.append(reward)

        # Discount rewards on every single game
        if reward == 1 or reward == -1:
            ep_rewards = np.vstack(ep_rewards_list)   # [0,0,..., -1]  -->    shape: (-1,1) e.g (84,1)
            discounted_rewards = discount_rewards(ep_rewards, gamma)
            rewards = np.vstack([rewards, discounted_rewards])
            ep_rewards_list = []
            # print(ep_rewards, discounted_rewards)
            #print("Ep reward {}".format(reward))
        if done:
            xs = np.vstack(xs_list)
            ys = np.vstack(ys_list)
            
            l,_ = sess.run([loss, train],
                               feed_dict={X: xs,
                                          Y: ys,
                                          advantages: rewards})
            reward_sum_history.append(reward_sum)
            break

    # Print status
    if global_step % 1 ==0:
        print("Average reward for last 100 episode {}: {:.3f} Loss: {:.6f}. elapsed: {} sec".format(global_step, np.mean(reward_sum_history), l,int(time.time()-s_time)))

    if global_step % 1 == 0:
        print("Saving network...")
        if not os.path.exists(CHECK_POINT_DIR):
            os.makedirs(CHECK_POINT_DIR)
        saver.save(sess, os.path.join(CHECK_POINT_DIR,CHECK_POINT_PREFACE), global_step=global_step)
