# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 01:12:01 2018

@author: Philip Docena
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import gym


def policy_pi_w(s,w):
    """ select action via softmax probability """
    a_probs=np.dot(s,w)
    a=np.argmax(np.exp(a_probs)/sum(np.exp(a_probs)))
    return a


def expected_return_sigma(gamma,r_ep):
    """ discount rewards via dynamic programming to avoid the large exponents """
    gamma_disc=np.zeros(len(r_ep))
    r_disc=r_ep[0]
    gamma_disc[0]=1
    for i in range(1,len(r_ep)):
        gamma_disc[i]=gamma*gamma_disc[i-1]
        r_disc+=gamma_disc[i]*r_ep[i]
    return r_disc


def hill_climb(w,w_best,J,J_best,alpha,rhc_epsilon,attempts):
    """ randomized hill climb """
    if J>=J_best:        # the nature of this problem requires lateral moves
        w_best=w
        J_best=J
        attempts=0
    else:
        attempts+=1
    
    # scale up weight change if failure; cap max
    rhc_epsilon=rhc_epsilon if J>=J_best else min(0.1,rhc_epsilon*2)
    w+=alpha*rhc_epsilon*np.random.rand(w.shape[0],w.shape[1])
    return w,w_best,J_best,rhc_epsilon,attempts


def main():
    cartpole_type='CartPole-v0'
    #cartpole_type='CartPole-v1'
    env=gym.make(cartpole_type)
    target_reward=195 if cartpole_type=='CartPole-v0' else 475

    env_seed,numpy_seed,env_testing_seed=0,1,123
    env.seed(env_seed)
    np.random.seed(numpy_seed)

    num_sensors,num_actions=4,env.action_space.n
    weight_random_init,rhc_epsilon=1e-4,1e-1
    max_train_iters,max_attempts,min_rollouts,test_iters=500,10,100,200
    alpha,gamma=0.001,0.99

    # model a linear combo parameter vector per action, i.e., a fully connected 4-node hidden layer
    w=np.random.rand(num_sensors,num_actions)*weight_random_init

    # training phase
    print('training to find an optimal policy')
    r_training=[]
    w_best,J_best=w,-np.inf
    num_rollouts=0

    while True:
        s1=env.reset()
        r_ep=[]
        attempts=0
        
        while True:
            #env.render()
            a=policy_pi_w(s1,w)
            s2,r,done,_=env.step(a)
            r_ep.append(r)
            
            if done: break
            s1=s2                
        
        r_training.append(np.sum(r_ep))
        
        J=expected_return_sigma(gamma,r_ep)
        w,w_best,J_best,rhc_epsilon,attempts=hill_climb(w,w_best,J,J_best,alpha*(1-num_rollouts/max_train_iters),rhc_epsilon,attempts)
            
        num_rollouts+=1
        if num_rollouts>max_train_iters:
            print('RHC reached maximum iterations of {}.'.format(max_train_iters))
            break
        if np.mean(r_training[-100:])>target_reward and num_rollouts>=min_rollouts:
            print('RHC reached the target reward of {} on episode {}.'.format(target_reward,num_rollouts))
            break
        if attempts>max_attempts:
            print('RHC could not find better weights after {} attempts on episode {}'.format(max_attempts,num_rollouts))
            break
            
    # testing phase
    print('testing final policy')
    env.seed(env_testing_seed)
    r_testing=[]
    #for i in range(test_iters):
    for i in range(max(num_rollouts,100)):
        s=env.reset()
        r_ep=0
        
        while True:
            #env.render()
            a=policy_pi_w(s,w_best)
            s,r,done,_=env.step(a)
            r_ep+=r
            
            if done: break 
        
        #print('episode: {}  reward: {:.2f}'.format(i,r_ep))
        r_testing.append(r_ep)
                
    print('average reward on testing',np.mean(r_testing))
    env.close()
    
    # plot results
    plt.plot(r_training,label='training, moving ave (100)')
    plt.plot(r_testing,label='testing, per episode')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('Randomized Hill Climb Policy Optimization on {}\n \
              env seed={}, training seed={}, testing seed={}\n \
              Philip Docena'.format(cartpole_type,env_seed,numpy_seed,env_testing_seed),fontsize=10)
    plt.legend(loc='best')
    plt.show()


if __name__=='__main__':
    main()