from JMDSystem import My_system
from models import Pol_Net, Policy_Net
from jax import random
import jax.numpy as jnp
import numpy as onp
import jax




def Batch_chooseindex_e_greedy(Mux_Muy,probs,key,B_sz,epsilon=1.0):#eps= 1.0 during training
    """Chooses n-nodes , one from each graph, after renormalizing the output probabilities
    """
    N=int(Mux_Muy.shape[0]/B_sz)
    node_indeces=onp.zeros((B_sz,),dtype='int32')
    node_probs=jnp.zeros((B_sz,1))
    choosen_Mu_vec=jnp.zeros((B_sz,2,))
    key1, key2 = random.split(key, 2)
    keys=random.split(key2,B_sz)
    sample_p = random.uniform(key1,(B_sz,))
    for p in range(len(sample_p)):
        myprobs=probs[p*N:(p+1)*N]
        myprobs=myprobs*(1/jnp.sum(myprobs))
        if(sample_p[p]<epsilon):
            #Greedy
            random_choice=random.choice(keys[p],myprobs,p=myprobs.reshape(-1))
        else:
            #Random
            random_choice=random.choice(keys[p],myprobs)  
        ind=int(jnp.where(myprobs==random_choice,size=1)[0][0])
        node_indeces[p]=ind
        node_probs=node_probs.at[p].set(myprobs[ind])
        choosen_Mu_vec=choosen_Mu_vec.at[p].set(Mux_Muy[p*N+ind])
    return node_indeces, choosen_Mu_vec, node_probs


#Disp Function
@jax.jit
def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)
    
    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* jnp.pi)**(len(mu)/2)) * (jnp.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(jnp.linalg.inv(cov))).dot((x-mu))
    #return part1 * jnp.exp(part2)
    return jnp.exp(part2)

#Disp Function
def Batch_pred_disp_vec(Mu,key,B_sz=1,std=0.01):
    mean = Mu
    cov = jnp.array([[[std**2, 0], [0, std**2]]]*B_sz)
    Pred_disp= jax.random.multivariate_normal(key,mean,cov)
    probs=[]
    for i in range(B_sz):
        probs+=[pdf_multivariate_gauss(Pred_disp[i].reshape(-1,1), mean[i].reshape(-1,1), cov[i])[0]]
    probs=jnp.array(probs).reshape(-1)
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
Sys=My_system()

#Load data
Train,Val,Test=Sys.create_batched_States(key ,N_sample=1500,Batch_size=10)


model=Pol_Net(edge_emb_size=16
    ,node_emb_size=16
    ,fa_layers=2
    ,fb_layers=2
    ,fv_layers=2
    ,fe_layers=2
    ,MLP1_layers=2
    ,MLP2x_layers=2
    ,MLP2y_layers=2
    ,train=True
    ,message_passing_steps=1)
    
#Initializing model parameters
key1, key2 = random.split(random.PRNGKey(147), 2)
params = model.init(key2, Train[0].Graph)

#loss function
def loss_fn(params,Systate,key,len_ep=5,Batch_id=1):
    log_length=len_ep
    B_sz=Systate.N.shape[0]
    Systate_temp=Systate
    apply_fn=model.apply
    #apply_fn=jax.jit(model.apply)
    log_length=len_ep
    log = {
        'Node_id': jnp.zeros((B_sz,log_length)),
        'Node_prob': jnp.zeros((B_sz,log_length)),
        'Mux_Muy': jnp.zeros((B_sz,log_length,2)),    
        'Disp_vec': jnp.zeros((B_sz,log_length,2)),
        'Disp_prob':jnp.zeros((B_sz,log_length,)),
        'Total_prob':jnp.zeros((B_sz,log_length,)),
        'Reward':jnp.zeros((B_sz,log_length,)),
        'd_PE':jnp.zeros((B_sz,log_length,)),
        'PE':jnp.zeros((B_sz,log_length,)),
        'States':[]}
    #Systate_temp=Systate
    for i in range(len_ep):
        key, key1,key2 = random.split(key, 3)
        #1: Pass through Policy_net
        #Batch_G, Batch_node_probs, Batch_Mux_Muy = apply_fn(params, Systate_temp.Graph)
        (Batch_G, Batch_node_probs, Batch_Mux_Muy),mutated_vars = apply_fn(params, Systate_temp.Graph,mutable=['batch_stats'])
        #print("Batch_G.nodes",Batch_G.nodes)
        #2: Choose node and disp from prob distributions
        Batch_chosen_node_index,Batch_chosen_Mux_Muy,Batch_chosen_node_prob=Batch_chooseindex_e_greedy(Batch_Mux_Muy,Batch_node_probs,key=key1,B_sz=B_sz,epsilon=1.0)#eps= 1.0 during training
        Batch_Disp_vec, Batch_log_prob,Batch_prob_disp= Batch_pred_disp_vec(Batch_chosen_Mux_Muy,key2,B_sz=B_sz,std=1e-3)
        Batch_log_node_prob=jnp.log(Batch_chosen_node_prob)
        Log_Pi_a_given_s=Batch_log_prob+Batch_log_node_prob.reshape(-1)

        #3: Displace chosen nodes with predicted displacement
        Systate_new,Batch_d_PE=Sys.multi_disp_node(Batch_chosen_node_index,Batch_Disp_vec,Systate_temp)
        Systate_temp=Systate_new
        log['d_PE']=log['d_PE'].at[:,i].set(Batch_d_PE)
        log['PE']=log['PE'].at[:,i].set(Systate_temp.pe)
        log['Node_id']=log['Node_id'].at[:,i].set(Batch_chosen_node_index)
        log['Node_prob']=log['Node_prob'].at[:,i].set(Batch_chosen_node_prob.reshape(-1))
        log['Mux_Muy']=log['Mux_Muy'].at[:,i].set(Batch_chosen_Mux_Muy)
        log['Disp_vec']=log['Disp_vec'].at[:,i].set(Batch_Disp_vec)
        log['Disp_prob']=log['Disp_prob'].at[:,i].set(jnp.exp(jnp.sum(Batch_log_prob)))
        log['Total_prob']=log['Total_prob'].at[:,i].set(jnp.exp(Log_Pi_a_given_s))
        log['Reward']=log['Reward'].at[:,i].set(-1*Batch_d_PE)
        log['States']+=[Systate_temp]
    loss_batch=Traj_Loss_fn(log_probs=jnp.log(log['Total_prob']),Returns=get_discounted_returns(log['Reward']))  #Shape: (B_sz,)
    #Taking sum of loss
    loss=jnp.sum(loss_batch)/B_sz
    return loss, (Systate_temp,log)    #Returns updated graph



#Defining optimizer
import optax
import flax
from flax.training import train_state
from flax import serialization
from flax.training import checkpoints as ckp
import matplotlib.pyplot as plt

#schedule = optax.warmup_cosine_decay_schedule(
#  init_value=1e-8,
#  peak_value=0.001,
#   warmup_steps=50,
#   decay_steps=500,
#   end_value=0.0,
# )

tx = optax.chain(
  optax.clip(0.5),
  optax.sgd(learning_rate=0.005)
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
Batch_size        =  2
len_ep            =  10
Print_freq        =  1
plot_freq         =  100
Model_save_freq   =  100
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
            ((val_loss_val,(val_Systate,val_log)),grads) = loss_grad_fn(params,Val[(Batch_size*i+p)%len(Val)],keys2[p],len_ep=len_ep)
            print_log(val_log,is_plot=False,epoch_id=i,Batch_id=p)
            Val_loss_data+=[Val_Batch_loss/Batch_size]
            Val_loss_epochs+=[i]
        
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
        ckp.save_checkpoint("./checkpoints/",params,i,overwrite=True,keep=4)
