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

from Systems import lennard_jones, SW_Silicon, CSH
import Systems
from Generic_system import Generic_system
from Utils import *
from Optimizers import *

        
#Defining model

key = random.PRNGKey(119)
Sys=Generic_system()

Train,Val,Test, shift_fn,Batch_pair_cutoffs,Batch_pair_sigma,Batch_Disp_Vec_fn,Batch_Node_energy_fn,Batch_Total_energy_fn,displacement_fn, shift_fn,Disp_Vec_fn =Sys.create_batched_States(random.PRNGKey(147),System='CSH',spatial_dimension=3,N=152, N_sample =100,Batch_size=4)    

def Total_energy_fn(R):
    return Batch_Total_energy_fn(R[jnp.newaxis,:,:])[0]
Total_energy_fn=jax.jit(Total_energy_fn)

model=Pol_Net(edge_emb_size=48
    ,node_emb_size=48
    ,fa_layers=2
    ,fb_layers=2
    ,fv_layers=2
    ,fe_layers=2
    ,MLP1_layers=1
    ,MLP2_layers=4
    ,spatial_dim=3
    ,sigma=5
    ,train=True
    ,message_passing_steps=1)
    
#Initializing model parameters
key1, key2 =random.split(random.PRNGKey(147), 2)
params = model.init(key2, Train[0].Graph)

#loss function
def loss_fn(params,Systate,key,spatial_dim=3,len_ep=10,Batch_id=1):
    len_ep=len_ep
    K_disp=152
    Batch_id=1
    log_length=len_ep
    B_sz=Systate.N.shape[0]
    Systate_temp=Systate
    apply_fn=model.apply
    #apply_fn=jax.jit(model.apply)
    log_length=len_ep
    log = {
        'Max_Mu': jnp.zeros((B_sz,log_length,)),    
        'Max_Disp': jnp.zeros((B_sz,log_length,)),
        'Mean_Mu': jnp.zeros((B_sz,log_length,)),    
        'Mean_Disp': jnp.zeros((B_sz,log_length,)),
        'Total_prob':jnp.zeros((B_sz,log_length,)),
        'Reward':jnp.zeros((B_sz,log_length,)),
        'd_PE':jnp.zeros((B_sz,log_length,)),
        'PE':jnp.zeros((B_sz,log_length,)),
        'States':[]}
    
    Test_R=[Systate_temp.R[k] for k in range(B_sz)]
    PE1=onp.zeros((len_ep,B_sz))
    for k in range(B_sz):
        PE1[:,k]=Batch_Total_energy_fn(FIRE_desc(1e-2,len_ep,Test_R[k],Total_energy_fn,shift_fn)[1])
    for i in range(len_ep):
        key, key1,key2 = random.split(key, 3)
    
        #1: Pass through Policy_net
        (Batch_G, Batch_node_probs, Batch_Mux_Muy),mutated_vars = apply_fn(params, Systate_temp.Graph,mutable=['batch_stats'])
        Batch_Mux_Muy=jnp.clip(Batch_Mux_Muy,-5,5)  #Limit displacements
        #2: Choose node and disp from prob distributions
        Batch_Disp_vec, Batch_log_disp_prob,Batch_prob_disp= Batch_pred_disp_vec(Batch_Mux_Muy.reshape(B_sz,-1,spatial_dim),key2,B_sz=B_sz,std=1e-6)
        
        Log_Pi_a_given_s=jnp.sum(Batch_log_disp_prob,axis=1)
        #3: Displace all nodes with predicted displacement
        Systate_new,Batch_d_PE=Sys.multi_disp_node(Batch_Disp_vec,Systate_temp,shift_fn,Batch_pair_cutoffs,Batch_pair_sigma,Batch_Disp_Vec_fn,Batch_Node_energy_fn,Batch_Total_energy_fn)
        log['d_PE']=log['d_PE'].at[:,i].set(Batch_d_PE)
        log['PE']=log['PE'].at[:,i].set(Systate_temp.pe)
        
        Mu_magnitude=jnp.sum(Batch_Mux_Muy**2,axis=1).reshape((B_sz,-1,))
        Disp_magnitude=jnp.sum(Batch_Disp_vec**2,axis=2)
        
        log['Max_Mu']=log['Max_Mu'].at[:,i].set(jnp.sqrt(jnp.max(Mu_magnitude,axis=1)))
        log['Mean_Mu']=log['Mean_Mu'].at[:,i].set(jnp.sqrt(jnp.mean(Mu_magnitude,axis=1)))
        
        log['Max_Disp']=log['Max_Disp'].at[:,i].set(jnp.sqrt(jnp.max(Disp_magnitude,axis=1)))
        log['Mean_Disp']=log['Mean_Disp'].at[:,i].set(jnp.sqrt(jnp.mean(Disp_magnitude,axis=1)))
        
        log['Total_prob']=log['Total_prob'].at[:,i].set(Log_Pi_a_given_s/100)
        log['Reward']=log['Reward'].at[:,i].set(-1*(Systate_new.pe-PE1[-1,:]))
        log['States']+=[Systate_temp]
        #Update current state
        Systate_temp=Systate_new
    
    loss_batch=Traj_Loss_fn(log_probs=log['Total_prob'],Returns=get_discounted_returns(log['Reward']))  #Shape: (B_sz,)
    #Taking mean of loss
    loss=jnp.sum(loss_batch)/B_sz
    return loss, (Systate_temp,log)    #Returns updated graph

#Defining optimizer
import optax
import flax
from flax.training import train_state
from flax import serialization
from flax.training import checkpoints as ckp
import matplotlib.pyplot as plt


tx = optax.chain(
  optax.clip(0.1),
  optax.adam(learning_rate=0.005)
)

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
len_ep            =  12
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
        print_log(log,epoch_id=i,Batch_id=p)
        Batch_loss+=loss_val
        grads_acc=add_grads(grads_acc,grads)
        
    if((i+1)%Val_freq==0):
        Val_Batch_loss=0
        for p in range(Batch_size):
            ((val_loss_val,(val_Systate,val_log)),grads) = loss_grad_fn(params,Val[(Batch_size*i+p)%len(Val)],keys2[p],len_ep=20)
            print_log(val_log,epoch_id=i,Batch_id=p)
            print("D_PE_sum:",jnp.sum(val_Systate.pe-Val[(Batch_size*i+p)%len(Val)].pe))
        
            Val_loss_data+=[Val_Batch_loss/Batch_size]
            Val_loss_epochs+=[i]
            Val_Batch_loss+=val_loss_val
        
    updates, opt_state = tx.update(scalar_mult_grad(1/Batch_size,grads_acc), opt_state,params)
    Train_loss_data+=[Batch_loss/Batch_size]
    Train_loss_epochs+=[i]
    params = optax.apply_updates(params, updates)
    if i % Print_freq == 0:
        if((i+1)%Val_freq==0):
            print('Val-Loss step {}: '.format(i), Val_Batch_loss/Batch_size)
        print('Loss step {}:{} '.format(i, Batch_loss/Batch_size))
    if(i%Model_save_freq==0):
        ckp.save_checkpoint("./checkpoints/",params,i,overwrite=True,keep=400)
