import numpy as onp
import scipy.stats as st
from jax.config import config  
config.update('jax_enable_x64', True)
from models import Pol_Net, Policy_Net

import jax
import jax.numpy as jnp
from jax import random,jit,lax
import jax
from jax import ops
from jax.example_libraries import optimizers
from jax.lax import fori_loop
import jraph
import time

import jax_md
from jax_md import space, smap, energy, minimize, quantity, simulate
from typing import Any, NamedTuple,  Optional, Union
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from jax._src import prng
Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]
import timeit
import matplotlib
import matplotlib.pyplot as plt

#Debugging
from Systems import lennard_jones, SW_Silicon, CSH
import Systems
from Generic_system import Generic_system
from Utils import *
from Optimizers import *




def get_index(elem,arr):
    return jnp.where(elem==arr,size=1)[0][0]
get_indeces_fn=jax.jit(jax.vmap(get_index,in_axes=(0,None)))


def Batch_choose_topK_e_greedy(Mux_Muy,probs,key,B_sz,K=1,epsilon=1.0):#eps= 1.0 during training
    """Chooses n-nodes*K nodes , K from each graph, after renormalizing the output probabilities
    """
    N=int(Mux_Muy.shape[0]/B_sz)
    node_indeces=onp.zeros((B_sz,K),dtype='int32')
    node_probs=jnp.zeros((B_sz,K))
    #choosen_Mu_vec=jnp.zeros((B_sz,N,2))
    key1, key2 = random.split(key, 2)
    keys=random.split(key2,B_sz)
    sample_p = random.uniform(key1,(B_sz,))
    for p in range(len(sample_p)):
        myprobs=probs[p*N:(p+1)*N]
        myprobs=myprobs*(1/jnp.sum(myprobs))
        if(sample_p[p]<epsilon):
            #Greedy
            random_choice=random.choice(keys[p],myprobs,shape=(K,1),replace=False,p=myprobs.reshape(-1))
        else:
            #Random
            random_choice=random.choice(keys[p],myprobs,shape=(K,1),replace=False)  
        ind=get_indeces_fn(random_choice,myprobs)
        node_indeces[p,:]=ind
        node_probs=node_probs.at[p,:].set(myprobs[ind].reshape(-1))
        #choosen_Mu_vec=choosen_Mu_vec.at[p,ind].set(Mux_Muy[p*N+ind])
    return node_indeces, node_probs

#Disp Function

           #Disp Function

def pdf_multivariate_gauss(x, mu, cov):
    """Removed part1 for scaling reason[w/0 part1 it is in 0 to 1]: pdf gives density not probability"""
   
    '''
    Caculate the multivariate normal density (pdf)
    
    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    #assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    #assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    #assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    #assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    #assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    #part1 = 1 / ( ((2* jnp.pi)**(len(mu)/2)) * (jnp.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(jnp.linalg.inv(cov))).dot((x-mu))
    #return part1 * jnp.exp(part2)
    return jnp.exp(part2)

vmap_pdf_multivariate_gauss=jax.jit(jax.vmap(jax.vmap(pdf_multivariate_gauss)))

def Batch_pred_disp_vec(Mu,key,B_sz=1,std=0.01,spatial_dim=3):
    mean = Mu
    K=mean.shape[1]
    cov = jnp.array([jnp.eye(spatial_dim)*(std**2)]*B_sz*K).reshape((B_sz,K,spatial_dim,spatial_dim))
    Pred_disp= jax.random.multivariate_normal(key,mean,cov)
    probs=vmap_pdf_multivariate_gauss(Pred_disp,mean,cov)
    return Pred_disp, jnp.log(probs), probs
    #Does not performs the dislpacement of node here, only predicts the node and displacement vector                       
          


#Discounted reward function
@jax.jit
def get_discounted_returns(Rewards,Y=0.9):
    """Calculates discounted rewards"""
    res=jnp.zeros(Rewards.shape)
    #res=Rewards
    Temp_G=onp.zeros((Rewards.shape[0],))
    for k in range(Rewards.shape[1]-1,-1,-1):
        Temp_G=Rewards[:,k]+Y*Temp_G
        res=res.at[:,k].set(Temp_G)
    return res
    
#Defining loss function
@jax.jit
def Traj_Loss_fn(*,log_probs, Returns):
    return jnp.sum(log_probs*Returns,axis=1)

#Grad_add function
@jax.jit
def add_grads(grad1,grad2):
    return jax.tree_multimap(lambda x,y:x+y,grad1,grad2)

@jax.jit
def scalar_mult_grad(k,grad):
    return jax.tree_map(lambda x:k*x,grad)

def print_log(log,is_plot=False,epoch_id=0,Batch_id=0):
    B_sz=log['Node_id'].shape[0]
    log_length=log['Node_id'].shape[1]
    if(is_plot==True):
        for i in range(len(log['States'])):
            Sys.plot_batched(log['States'][i],epoch_id=epoch_id,batch_id=Batch_id,step_id=i,node_ids=log['Node_id'][i],Edges=True,save=False)
            #Sys.plot_frame_edge(ax1,log['States'][i],node_id=int(log['Node_id'][i]))
    for k in range(B_sz):
        print("\n#GraphNo. ",k+1)
        print("\nN_id\tNode_prob\tMux Muy\t\tDisp_x Disp_y\tDisp_prob\tTotal_prob\tReward  d_PE  PE")
        for i in range(log['Node_id'].shape[1]):
            print("%d\t%8.6f  %8.6f %8.6f\t%8.6f %8.6f   %8.6f\t%8.6f %8.5f  %8.5f  %8.5f"%(log['Node_id'][k][i],log['Node_prob'][k][i],log['Mux_Muy'][k][i][0],log['Mux_Muy'][k][i][1],log['Disp_vec'][k][i][0],log['Disp_vec'][k][i][1],log['Disp_prob'][k][i],log['Total_prob'][k][i],log['Reward'][k][i],log['d_PE'][k][i],log['PE'][k][i]))
        

#Defining model

#from JMDSystem import MDTuple, My_system
key = random.PRNGKey(119)
Sys=Generic_system()

#Load data
Train,Val,Test, shift_fn,Batch_pair_cutoffs,Batch_pair_sigma,Batch_Disp_Vec_fn,Batch_Node_energy_fn,Batch_Total_energy_fn=Sys.create_batched_States(random.PRNGKey(147),System='LJ',spatial_dimension=3,N=100, N_sample =100,Batch_size=4)        


def Total_energy_fn(R):
    return Batch_Total_energy_fn(R[jnp.newaxis,:,:])[0]
Total_energy_fn=jax.jit(Total_energy_fn)

model=Pol_Net(edge_emb_size=32
    ,node_emb_size=32
    ,fa_layers=2
    ,fb_layers=2
    ,fv_layers=2
    ,fe_layers=2
    ,MLP1_layers=2
    ,MLP2_layers=3
    ,spatial_dim=3
    ,sigma=2.0
    ,train=True
    ,message_passing_steps=1)
    
#Initializing model parameters
key1, key2 = random.split(random.PRNGKey(147), 2)
params = model.init(key2, Train[0].Graph)


#loss function
def loss_fn(params,Systate,key,spatial_dim=3,len_ep=5,Batch_id=1):
    len_ep=len_ep
    K_disp=100
    Batch_id=1
    log_length=len_ep
    B_sz=Systate.N.shape[0]
    Systate_temp=Systate
    apply_fn=model.apply
    #apply_fn=jax.jit(model.apply)
    log_length=len_ep
    log = {
        'Node_id': jnp.zeros((B_sz,log_length)),
        'Node_prob': jnp.zeros((B_sz,log_length)),
        'Mux_Muy': jnp.zeros((B_sz,log_length,spatial_dim)),    
        'Disp_vec': jnp.zeros((B_sz,log_length,spatial_dim)),
        'Disp_prob':jnp.zeros((B_sz,log_length,)),
        'Total_prob':jnp.zeros((B_sz,log_length,)),
        'Reward':jnp.zeros((B_sz,log_length,)),
        'd_PE':jnp.zeros((B_sz,log_length,)),
        'PE':jnp.zeros((B_sz,log_length,)),
        'States':[]}
    #Systate_temp=Systate
    Test_R=[Systate_temp.R[k] for k in range(B_sz)]
    PE1=onp.zeros((len_ep,B_sz))
    for k in range(B_sz):
        PE1[:,k]=Batch_Total_energy_fn(FIRE_desc(1e-2,len_ep,Test_R[k],Total_energy_fn,shift_fn)[1])
    for i in range(len_ep):
        key, key1,key2 = random.split(key, 3)
    
        #1: Pass through Policy_net
        #Batch_G, Batch_node_probs, Batch_Mux_Muy = apply_fn(params, Systate_temp.Graph)
        (Batch_G, Batch_node_probs, Batch_Mux_Muy),mutated_vars = apply_fn(params, Systate_temp.Graph,mutable=['batch_stats'])
        #print("Batch_G.nodes",Batch_G.nodes)
    
        #2: Choose node and disp from prob distributions
        Batch_chosen_node_indeces,Batch_chosen_node_prob=Batch_choose_topK_e_greedy(Batch_Mux_Muy,Batch_node_probs,key=key1,B_sz=B_sz,K=K_disp,epsilon=1.0)#eps= 1.0 during training
        #To choose k nodes
        
        Node_mask=jnp.zeros(Batch_Mux_Muy.reshape(B_sz,-1,spatial_dim).shape)
        Node_mask=Node_mask.at[jnp.array([[i for i in range(Batch_chosen_node_indeces.shape[0])]]* Batch_chosen_node_indeces.shape[1]).T, Batch_chosen_node_indeces,:].set(1.0)
        Batch_Disp_vec, Batch_log_disp_prob,Batch_prob_disp= Batch_pred_disp_vec(Batch_Mux_Muy.reshape(B_sz,-1,spatial_dim),key2,B_sz=B_sz,std=1e-5)
        Batch_log_node_prob=jnp.log(Batch_chosen_node_prob)
        #print(Batch_prob_disp)
        #print(Batch_log_prob,Batch_log_node_prob)
        #print(Batch_chosen_Mux_Muy-Batch_Disp_vec)
        #print(jnp.sum(Batch_log_disp_prob),jnp.sum(Batch_log_node_prob))
    
        #print(jnp.sum(Batch_log_disp_prob+Batch_log_node_prob))
        #print(jnp.sum(Batch_log_disp_prob*Node_mask[:,:,0],axis=1).shape)
        Log_Pi_a_given_s=jnp.sum(Batch_log_disp_prob*Node_mask[:,:,0],axis=1)+jnp.sum(Batch_log_node_prob,axis=1)
        #3: Displace all nodes with predicted displacement
    #     Batch_Disp_vec_pred=Batch_Mux_Muy.at[Batch_chosen_node_indeces].set(Batch_Disp_vec)
    #     Node_indeces=jnp.array([[i for i in range(Systate.R.shape[1])] for k in range(B_sz)])
        Systate_new,Batch_d_PE=Sys.multi_disp_node(Batch_Disp_vec*Node_mask,Systate_temp,shift_fn,Batch_pair_cutoffs,Batch_pair_sigma,Batch_Disp_Vec_fn,Batch_Node_energy_fn,Batch_Total_energy_fn)
        Systate_temp=Systate_new
        log['d_PE']=log['d_PE'].at[:,i].set(Batch_d_PE)
        log['PE']=log['PE'].at[:,i].set(Systate_temp.pe)
        #log['Node_id']=log['Node_id'].at[:,i].set(Batch_chosen_node_index)
        #log['Node_prob']=log['Node_prob'].at[:,i].set(Batch_chosen_node_prob.reshape(-1))
        #log['Mux_Muy']=log['Mux_Muy'].at[:,i].set(Batch_chosen_Mux_Muy)
        #log['Disp_vec']=log['Disp_vec'].at[:,i].set(Batch_Disp_vec)
        #log['Disp_prob']=log['Disp_prob'].at[:,i].set(jnp.exp(jnp.sum(Batch_log_prob)))
        log['Total_prob']=log['Total_prob'].at[:,i].set(jnp.exp(Log_Pi_a_given_s))
        log['Reward']=log['Reward'].at[:,i].set(-1*(Systate_new.pe-PE1[-1,:]))
        log['States']+=[Systate_temp]
    loss_batch=Traj_Loss_fn(log_probs=jnp.log(log['Total_prob']),Returns=get_discounted_returns(log['Reward']))  #Shape: (B_sz,)
    #Taking sum of loss
    loss=jnp.sum(loss_batch)/B_sz
    if loss>1e15:
        loss=1e15
    elif loss<-1e15:
        loss=-1e15
    return loss, (Systate_temp,log)    #Returns updated graph

    
#Defining optimizer
import optax
import flax
from flax.training import train_state
from flax import serialization
from flax.training import checkpoints as ckp
import matplotlib.pyplot as plt

#config.update('jax_disable_jit', True)
#config.update("jax_debug_nans", True)

#schedule = optax.warmup_cosine_decay_schedule(
#  init_value=1e-8,
#  peak_value=0.001,
#   warmup_steps=50,
#   decay_steps=500,
#   end_value=0.0,
# )

tx = optax.chain(
  optax.clip(0.5),
  optax.sgd(learning_rate=0.004)
)
#tx=flax.optim.momentum(learning_rate=1e-3,beta=0.9,weight_decay=1,nestrov=True)

#tx = optax.sgd(learning_rate=0.001)
#Add l2 regularizer
#print(params)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

#Initial loss and gradients computation
((loss,(Systate_init,log_init)),init_grad)=loss_grad_fn(params,Train[0],random.PRNGKey(147))
print_log(log_init)
print("Initial Loss",loss)


#Training and Validation

Train_loss_data   =  []
Train_loss_epochs =  []
Val_loss_data     =  []
Val_loss_epochs   =  []
Batch_size        =  3
len_ep            =  10
Print_freq        =  1
plot_freq         =  100
Model_save_freq   =  10
N_epoch           =  800
Val_freq          =  20
keys1=random.split(random.PRNGKey(117721817),N_epoch)
for i in range(N_epoch):
    grads_acc=scalar_mult_grad(0.0,init_grad)
    keys2=random.split(keys1[i],Batch_size)
    
        
    Batch_loss=0
    for p in range(Batch_size):
        ((loss_val,(Systate,log)),grads) = loss_grad_fn(params,Train[(Batch_size*i+p)%len(Train)],keys2[p],len_ep=len_ep)
        print("Batch: ",p,' ',loss_val)
        print_log(log,is_plot=False,epoch_id=i,Batch_id=p)
        Batch_loss+=loss_val
        grads_acc=add_grads(grads_acc,grads)
    
    if((i+1)%Val_freq==0):
        Val_Batch_loss=0
        for p in range(Batch_size):
            ((val_loss_val,(val_Systate,val_log)),grads) = loss_grad_fn(params,Val[(Batch_size*i+p)%len(Val)],keys2[p],len_ep=20)
            print_log(val_log,is_plot=False,epoch_id=i,Batch_id=p)
            print("D_PE_sum:",jnp.sum(val_Systate.pe-Val[(Batch_size*i+p)%len(Val)].pe))
        
            Val_loss_data+=[Val_Batch_loss/Batch_size]
            Val_loss_epochs+=[i]
            Val_Batch_loss+=val_loss_val
        
    updates, opt_state = tx.update(scalar_mult_grad(1/Batch_size,grads_acc), opt_state,params)
    Train_loss_data+=[Batch_loss/Batch_size]
    Train_loss_epochs+=[i]
    old_params=params
    #print("myGrads:",scalar_mult_grad(1/Batch_size,grads_acc))
    #print("Before update",params)
    params = optax.apply_updates(params, updates)
    #print("After_update",params)
    #print(jax.tree_multimap(lambda x,y:100*(x-y)/y,params,old_params))
    if i % Print_freq == 0:
        if((i+1)%Val_freq==0):
            print('Val-Loss step {}: '.format(i), Val_Batch_loss/Batch_size)
        print('Loss step {}:{} '.format(i, Batch_loss/Batch_size))
        print('P.E {}: '.format(Systate.pe))
        #fig,ax=plt.subplots(1,1,figsize=(10,10))
        #Sys.plot_frame_edge(ax,Systate)
        #fig.savefig("./Plots/System_"+str(i)+"_"+str(Systate.pe)+"_plot"+".png")
        #plt.close(fig)
    if(i%Model_save_freq==0):
        ckp.save_checkpoint("./checkpoints/",params,i,overwrite=True,keep=400)




