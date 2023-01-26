import jax
import jax.numpy as jnp
import numpy as onp
from jax import random,jit,lax

@jax.jit
def _behler_parrinello_cutoff_fn(dr, cutoff_distance = 8.0):
    """Function of pairwise distance that smoothly goes to zero at the cutoff."""
    return jnp.where((dr < cutoff_distance) & (dr > 1e-7),0.5 * (jnp.cos(jnp.pi * dr / cutoff_distance) + 1), 0)


def radial_symmetry_functions(etas,
                              cutoff_distance,
                              num_species = 1):
    """Returns a function that computes radial symmetry functions.

        This is a re-implementation of the radial symmetry functions within
        nn.py but allows for vmapping by making use of input masking rather than
        array indexing.

        Args:
        metric_mapped: displacement function that computes distances in
          spatial positions between all pairs of atoms
        etas: [num_etas], list corresponding to strength of interaction terms
        cutoff_distance: neighbors whose distance is larger than cutoff_distance do
          not contribute to each others symmetry functions. The contribution of a
          neighbor to the symmetry function and its derivative goes to zero at this
          distance. [0.0009, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.4]
        num_species: total number of species

        Returns:
        A function that computes the radial symmetry fucntions from inputs, yielding
        output of shape [num_atoms,num_etas* num_species] to maintain the type
        consistency from nn.py
        """
    def radial_fn(eta, dr):
        cutoffs = _behler_parrinello_cutoff_fn(dr, cutoff_distance)
        return jnp.exp(-eta * dr**2) * cutoffs

    @jax.jit
    def compute_fun(dr, species):
        def return_radial(atom_type):
            mask = species == atom_type
            radial = jax.vmap(radial_fn, (0, None))(etas, dr)
            radial_masked = radial * mask.reshape([1, -1, 1])
            return jnp.sum(radial_masked, axis=-1)

        radial_vmap = jax.vmap(return_radial)
        radial_symmetry = radial_vmap(jnp.arange(num_species))
        radial_symmetry = jnp.transpose(radial_symmetry, (2, 0, 1))
        return radial_symmetry.reshape([radial_symmetry.shape[0], -1])

    return compute_fun




#Utilities
def matrix_index_fn(matrix,type_a,type_b):
    return matrix[type_a,type_b]


matrix_broadcast_fn=jax.jit(jax.vmap(jax.vmap(matrix_index_fn,(None,0,None)),(None,None,0)))
#To broadscast pair potential parameters to shape (N,N) from (N_species,N_species)
#Example usage pair_sigma=matrix_broadcast_fn(sigma,species,species) where species is (N,) array of species of each atom


def write_xyz(Filepath,Name,R,species):
    '''Writes ovito xyz file'''

    f=open(Filepath,'w')
    f.write(str(R.shape[0])+"\n")
    f.write(Name)
    for i in range(R.shape[0]):
        f.write("\n"+str(species[i])+"\t"+str(R[i,0])+"\t"+str(R[i,1])+"\t"+str(R[i,2]))

def write_xyz_traj(Filepath,species,Traj,box_size,Node_energies):
    '''Writes ovito xyz file
    Filepath        -Directory for writing
    N_species       -No. of species
    Species         -(N_Atoms, ) species identifier
    Traj            -Trajectory of shape (Nframes,N_atoms,spatial_dim)
    box_size        -scalar(1,)
    Node_energies   -pe of atoms (Nframes,N_Atoms,spatial_dim)
    '''
    N_frames=Traj.shape[0]
    N_atoms=Traj.shape[1]
    spatial_dim=Traj.shape[2]
    f=open(Filepath,'w')
    for frame_i in range(N_frames): 
        if(frame_i==0):
            f.write(str(N_atoms))    
        else:
            f.write("\n"+str(N_atoms))
        f.write("\nLattice=\""+str(box_size)+"0.0 0.0 0.0 "+str(box_size)+" 0.0 0.0 0.0 "+str(box_size)+"\" Properties=species:S:1:pos:R:"+str(spatial_dim)+":local_energy:R:1 Time="+str(frame_i))
        for i in range(R.shape[0]):
            f.write("\n"+str(species[i])+"\t"+str(Traj[frame_i,i,0])+"\t"+str(Traj[frame_i,i,1])+"\t"+str(Traj[frame_i,i,2])+"\t"+str(Node_energies[frame_i,i]))



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
def pdf_multivariate_gauss(x, mu, cov):
    """Removed part1 for scaling reason[w/0 part1 it is in 0 to 1]: pdf gives density not probability"""
    '''
    Caculate the multivariate normal density (pdf)
    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    #part1 = 1 / ( ((2* jnp.pi)**(len(mu)/2)) * (jnp.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(jnp.linalg.inv(cov))).dot((x-mu))
    return jnp.exp(part2)

vmap_pdf_multivariate_gauss=jax.jit(jax.vmap(jax.vmap(pdf_multivariate_gauss)))

def Batch_pred_disp_vec(Mu,key,B_sz=1,std=0.01,spatial_dim=3):
    mean = Mu
    K=mean.shape[1]
    cov = jnp.array([jnp.eye(spatial_dim)*(std**2)]*B_sz*K).reshape((B_sz,K,spatial_dim,spatial_dim))
    Pred_disp= jax.random.multivariate_normal(key,mean,cov)
    probs=vmap_pdf_multivariate_gauss(Pred_disp,mean,cov)
    return Pred_disp, jnp.log(probs), probs
          


#Discounted reward function
@jax.jit
def get_discounted_returns(Rewards,Y=0.9):
    """Calculates discounted rewards"""
    res=jnp.zeros(Rewards.shape)
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
    return jax.tree_map(lambda x,y:x+y,grad1,grad2)

@jax.jit
def scalar_mult_grad(k,grad):
    return jax.tree_map(lambda x:k*x,grad)

def print_log(log,epoch_id=0,Batch_id=0):
    B_sz=log['Reward'].shape[0]
    log_length=log['Reward'].shape[1]
    for k in range(B_sz):
        print("\n#GraphNo. ",k+1)
        print("\nStep\tMax_Mu\tMean_Mu\t   Max_Disp  Mean_Disp\tLog_Total_prob\tReward\t d_PE\t  PE")
        for i in range(log['Reward'].shape[1]):
            print(i+1,"\t%8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f  %8.5f"%(log['Max_Mu'][k][i],log['Mean_Mu'][k][i],log['Max_Disp'][k][i],log['Mean_Disp'][k][i],log['Total_prob'][k][i],log['Reward'][k][i],log['d_PE'][k][i],log['PE'][k][i]))
