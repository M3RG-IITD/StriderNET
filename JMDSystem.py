import numpy as onp
import scipy.stats as st
from jax.config import config  
config.update('jax_enable_x64', True)

import jax.numpy as jnp
from jax import random,jit,lax
from LMPSTrajReader import Lammps_Traj
import jax
from jax import ops
from jax.example_libraries import optimizers
from jax.lax import fori_loop
import jraph
import time

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
#config.update('jax_disable_jit', True)
#config.update("jax_debug_nans", True)

#[N,box_size,PE,species,R.sigma, epsilon, cutoffs]
class MDTuple(NamedTuple):
    N          : jnp.ndarray    #No. of atoms
    box_size   : jnp.ndarray    #Size of box [2d array of [[xlo, xhi],[ylo,yhi]]
    pe         : jnp.ndarray    #potential energy
    species    : jnp.ndarray    #Atom types list [Nx1]
    R          : jnp.ndarray    #Position vectors of atoms [Nx2]
    neigh_list : jnp.ndarray   #neigh list
    Graph      : Optional[jraph.GraphsTuple]   #Graph structure


class My_system:
    """Defines A65B35 2D LJ System with functionality to displace chosen atoms"""
    def create_batched_States(self,key :KeyArray,N_sample=180,Batch_size=10):
        """
        -Creates Train, Val, and Test data of batched system_states along with batched graphs and random shuffling
        -Current split is 60:20:20
        -
        """
        key1, key2 = jax.random.split(key,2)
        N=38
        sigma=onp.array([[1.0 ,0.8],[0.8 ,0.88]],dtype=onp.float32)
        epsilon=onp.array([[1.0 ,1.5],[1.5 ,0.5]],dtype=onp.float32)
        cutoffs=onp.array([[1.5 ,1.25],[1.25 ,2.0]],dtype=onp.float32)
        
        G_indeces=jax.random.permutation(key, onp.arange(0,N_sample-Batch_size+1,Batch_size))    
        Systates=[]
        for k in G_indeces:
            #Initialize variables for Systate
            Batch_N=onp.ones((Batch_size,))*N
            Batch_pe=onp.zeros((Batch_size,))
            Batch_box_size=onp.zeros((Batch_size,2,2))
            Batch_species=onp.zeros((Batch_size,N),dtype='int32')
            Batch_R=onp.zeros((Batch_size,N,2))
            Batch_neigh_list=onp.zeros((Batch_size,N,N),dtype=onp.int32)
            Batch_Graphs_list=[]  

            #Shuffle indeces within batch
            key3, key1 = jax.random.split(key1,2)
            Batch_indeces=jax.random.permutation(key3, onp.arange(k,k+Batch_size,1))
            counter=-1
            for ind in Batch_indeces:
                counter=counter+1
                #Now create a graph
                #1: Read MD Traj Data
                Filename="Dataset_MD/Combined/md_cool."+str(ind)+".lammpstrj"
                MDFrame=Lammps_Traj(Filename)
                myTraj=MDFrame.getTraj()[0]
                
                box_size=onp.array([[myTraj['xlo'],myTraj['xhi']],[myTraj['ylo'],myTraj['yhi']]])
                L=jnp.array([myTraj['xhi']-myTraj['xlo'],myTraj['yhi']-myTraj['ylo']])
                mydata :pd.DataFrame=myTraj.get('Data')                                 
                sorted_data=mydata.sort_values(by='id').to_numpy() 
                species=onp.add(sorted_data[:,1],-1).astype('int32')
                R=sorted_data[:,2:4]
                
                Batch_species[counter]=species
                Batch_R[counter]=R
                Batch_box_size[counter]=box_size
                G,Energy,n_list=self.create_G(R,L,species)
                Batch_neigh_list[counter]=n_list
                Batch_pe[counter]=Energy
                print(Energy)
                Batch_Graphs_list+=[G] 
                #Node_features+=[np.array(Neigh_Edge_pe_feats)]

            Systates+=[MDTuple(N=jnp.array(Batch_N),box_size=jnp.array(Batch_box_size),pe=jnp.array(Batch_pe),species=jnp.array(Batch_species),R=jnp.array(Batch_R),neigh_list=jnp.array(Batch_neigh_list), Graph=jraph.batch(Batch_Graphs_list))]
        key4, key5 = jax.random.split(key2)
        N_sample=int(N_sample/Batch_size)
        Test_indeces=onp.arange(int(0.8*N_sample),N_sample)
        Val_indeces=jax.random.permutation(key4, onp.arange(int(0.6*N_sample),int(0.8*N_sample)))
        Train_indeces=jax.random.permutation(key5, onp.arange(0,int(0.6*N_sample)))        

        Test_Batch=[]
        for ind in Test_indeces:
            Test_Batch+=[Systates[ind]]
        
        Val_Batch=[]
        for ind in Val_indeces:
            Val_Batch+=[Systates[ind]]
         
        Train_Batch=[]
        for ind in Train_indeces:
            Train_Batch+=[Systates[ind]]
        
        return Train_Batch,Val_Batch,Test_Batch

    #Create_graph_function
    def create_G(self,R,L,species):
        """
        R: Node Positions
        L: Box length in x and y
        species: Node type info 0 and 1

         """
        pair_dist_fn=jax.jit(jax.vmap(self.dist,(0,None,None)))
    
        #1: Calculate pair distances
        R_pair=pair_dist_fn(R,R,L)
        dr2_pair=jnp.sum(jnp.square(R_pair),axis=2)
        dr2_pair=dr2_pair.at[jnp.diag_indices_from(dr2_pair)].set(1e-3)  #To remove nan error in sqrt
        dr_pair=jnp.sqrt(dr2_pair)
            
        
        #2: setting up potential parameters and cutoffs
        N=R.shape[0]
        Na=int(round(0.65*N))
        Nb=int(round(0.35*N))
        sigma=jnp.block([[jnp.ones((Na,Na))*1.0 ,jnp.ones((Na,Nb))*0.8],[jnp.ones((Nb,Na))*0.8 ,jnp.ones((Nb,Nb))*0.88]])
        epsilon=jnp.block([[jnp.ones((Na,Na))*1.0 ,jnp.ones((Na,Nb))*1.5],[jnp.ones((Nb,Na))*1.5 ,jnp.ones((Nb,Nb))*0.5]])
        cutoffs=jnp.block([[jnp.ones((Na,Na))*1.5 ,jnp.ones((Na,Nb))*1.25],[jnp.ones((Nb,Na))*1.25 ,jnp.ones((Nb,Nb))*2.0]])

        #3: Creating neigh_list and senders and receivers
        n_list=(dr_pair<cutoffs).astype(int)
        n_list=n_list.at[jnp.diag_indices_from(n_list)].set(0)
        (senders,receivers)=jnp.where(n_list==1)

        #4: Calculating Energy of each pair of atoms
        E_pair=energy.lennard_jones(dr_pair,sigma,epsilon)
        E_pair=E_pair.at[jnp.diag_indices_from(E_pair)].set(0.0) #Removing any self energy 

        #5: Node features
        node_pe=jnp.sum(E_pair,0).reshape((-1,1))
        #broadcasting node_pe to N dim
        pair_node_pe=jnp.concatenate([node_pe]*N,axis=1).T
        neigh_pe_dist=pair_node_pe*n_list
        deg_node=jnp.sum(n_list,axis=1).reshape((-1,1))
        neigh_pe_sum=jnp.sum(neigh_pe_dist,axis=1).reshape((-1,1))/N
        neigh_pe_mean=neigh_pe_sum/deg_node
        #Node_feats=jnp.concatenate([species.reshape((-1,1)),node_pe,neigh_pe_sum,neigh_pe_mean],axis=1)
        Node_feats=jnp.concatenate([node_pe,neigh_pe_sum,neigh_pe_mean],axis=1)

        #6: Edge Features
        dist_2d=R_pair[senders,receivers,:]
        l_sigma=(dr_pair-sigma)[senders,receivers].reshape((-1,1))
        Edge_feats=jnp.concatenate([dist_2d,l_sigma],axis=1)
        G=jraph.GraphsTuple(nodes=Node_feats,
                                    edges=Edge_feats,
                                    senders=senders,
                                    receivers=receivers,
                                    n_node=jnp.array([Node_feats.shape[0]]), 
                                    n_edge=jnp.array([Edge_feats.shape[0]]),globals=None)
        
        Energy=jnp.sum(node_pe)/2
        return G, Energy, n_list
    
    
    
    def dist(self,Ra,R,L):
        """
        -Calculates distance between central position 'Ra' and vector of postitions R, considering periodic boundaries
        -Ra: Central position
        -R: Vector of postitions
        -L: Box lengths (2,) for 2D
        """
        return ((R-Ra)+L/2)%L-L/2
    
    def lj(self,r,sigma,eps):
        """
        -Calculatrs lennard jones potential energy
        """
        a=jnp.power((sigma/r),6)
        return 4*eps*a*jnp.add(a,-1)
        
    
    def finalize_plot(self,shape=(1, 1)):
        """Helper function for size of plots"""
        plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
        plt.tight_layout()
    
    def plot_batched(self,Systate: MDTuple,epoch_id=0,batch_id=0,step_id=0,node_ids=None,Edges=True,save=False):
        """To plot batched system state"""
        ms=65
        N,box_size,pe,species,R,n_adj,G=Systate
        G_list=jraph.unbatch(G)  #Unbatch graphs to list of graphs
        for k in range(len(G_list)):
            fig,ax=plt.subplots(1,1,figsize=(10,8))
            G=G_list[k]
            R_plt=R[k]
            colors=G.nodes[:,1]
            senders=G.senders
            receivers=G.receivers
            v_min=np.round(np.mean(colors)-2*np.std(colors),decimals=1)
            v_max=np.round(np.mean(colors)+2*np.std(colors),decimals=1)
            cmap=plt.cm.seismic
            ax.scatter(R_plt[:, 0], R_plt[:, 1], s=(jnp.add(species[k],1)*ms*10),c=colors,cmap=cmap,norm=Normalize(v_min,v_max),edgecolors='k')
            cbar=plt.colorbar(ScalarMappable(Normalize(v_min,v_max),cmap=cmap))
            cbar.set_label("PE",weight='bold',rotation=270)
            if(node_ids!=None):
                ax.plot(R_plt[int(node_ids[k]), 0], R_plt[int(node_ids[k]), 1], 'o',color='g', markersize=ms *0.2)
            if(Edges==True):
                for m in range(0,senders.shape[0]-1,1):
                    R_0=R_plt[senders[m]]
                    R_1=R_plt[receivers[m]]
                    ax.plot([R_0[0],R_1[0]],[R_0[1],R_1[1]],color='k')
            ax.set_xlim([box_size[k,0,0], box_size[k,0,1]])
            ax.set_ylim([box_size[k,1,0], box_size[k,1,1]])
            ax.set_title("System_Epoch_"+str(epoch_id)+"_Batch_"+str(batch_id)+"_Step_"+str(step_id)+"_G_"+str(k)+"_PE_"+str(Systate.pe[k]))
            if(save==False):
                plt.show()
            else:
                fig.savefig("./Plots/System_Epoch_"+str(epoch_id)+"_Batch_"+str(batch_id)+"_Step_"+str(step_id)+"_G_"+str(k)+"_PE_"+str(Systate.pe[k])+"_plot"+".png")
                plt.close(fig)
        
        return None
    
    def plot_frame_edge(self,ax,Systate :MDTuple,node_id=None,Edges=True,color_pe=True):
        """Plots single graph state"""
        ms=65
        N,box_size,pe,species,R,n_adj,G=Systate
        senders=G.senders
        R_plt = R
        colors=[]
        if(color_pe==True):
            colors=G.nodes[:,2]
            v_min=np.round(np.mean(colors)-2*np.std(colors),decimals=1)
            v_max=np.round(np.mean(colors)+2*np.std(colors),decimals=1)
            cmap=plt.cm.seismic
            ax.scatter(R_plt[:, 0], R_plt[:, 1], s=(jnp.add(species,1)*ms*10),c=colors,cmap=cmap,norm=Normalize(v_min,v_max),edgecolors='k')
            cbar=plt.colorbar(ScalarMappable(Normalize(v_min,v_max),cmap=cmap))
            cbar.set_label("PE",weight='bold',rotation=270)
    
        else:
            for k in species:
                if(k==0):
                    colors+=['b']
                else:
                    colors+=['r']
            ax.scatter(R_plt[:, 0], R_plt[:, 1], s=(jnp.add(species,1)*ms*10),c=colors)
        
        if(node_id!=None):
            ax.plot(R_plt[node_id, 0], R_plt[node_id, 1], 'o',color='g', markersize=ms *0.2)
        if(Edges==True):
            for k in range(0,senders.shape[0]-1,2):
                R_0=R_plt[senders[k]]
                R_1=R_plt[senders[k+1]]
                ax.plot([R_0[0],R_1[0]],[R_0[1],R_1[1]],color='k')
        ax.set_xlim([0, jnp.max(R[:, 0])])
        ax.set_ylim([0, jnp.max(R[:, 1])])
        ax.set_title("Systate PE: "+str(Systate.pe))
    
    def multi_disp_node(self,Node_indeces,Disp_Vecs,Systate :MDTuple):
        """
        -To displace several nodes at once and return updated batched system state and change in PE
        -Node_indeces: Index of nodes to displace (0-37)
        -Disp_Vecs   : Displacement vectors of nodes to displace
        -Systate     : Current system state
        """
        #1: Unwrap system state
        Batch_N,Batch_box_size,Batch_PE_initial,Batch_species,Batch_R,Batch_neigh_list,Batch_OldG=Systate
        Batch_size=len(Batch_N)
        #2: setting up potential parameters and cutoffs
        N=38   #=Batch_R.shape[1]
        Na=int(round(0.65*N))
        Nb=int(round(0.35*N))
        sigma=jnp.block([[jnp.ones((Na,Na))*1.0 ,jnp.ones((Na,Nb))*0.8],[jnp.ones((Nb,Na))*0.8 ,jnp.ones((Nb,Nb))*0.88]])
        epsilon=jnp.block([[jnp.ones((Na,Na))*1.0 ,jnp.ones((Na,Nb))*1.5],[jnp.ones((Nb,Na))*1.5 ,jnp.ones((Nb,Nb))*0.5]])
        cutoffs=jnp.block([[jnp.ones((Na,Na))*1.5 ,jnp.ones((Na,Nb))*1.25],[jnp.ones((Nb,Na))*1.25 ,jnp.ones((Nb,Nb))*2.0]])
        pair_dist_fn=jax.jit(jax.vmap(self.dist,(0,None,None)))

        #3 : Displace the node's position vector
            #Currently assumes box boundaries starts at origin
        xlo=0.0
        ylo=0.0
        for p in range(len(Node_indeces)):
            box_size=Batch_box_size[p]
            L=jnp.array([box_size[0,1]-box_size[0,0],box_size[1,1]-box_size[1,0]])
            cur_pos=Batch_R[p,Node_indeces[p],:]  #Shape=(Batch_size,Batch_size,2)
            Batch_R=Batch_R.at[p,Node_indeces[p],0].set(xlo+(cur_pos[0]-xlo+Disp_Vecs[p,0])%L[0])
            Batch_R=Batch_R.at[p,Node_indeces[p],1].set(ylo+(cur_pos[1]-ylo+Disp_Vecs[p,1])%L[1])         
        
        """Now to update G_tuple"""
        #Node_feat=[Node_type(one-hot),Node_PE,Neigh_PE_mean, Neigh_PE_sum, Neigh_PE_median]
        #Edge_feat=[2D Edge vector,l-sigma] (#Edge_type(one-hot) not included, since node_type already given
        
       
        #Initialize variables for Systate
        Batch_pe=jnp.zeros((Batch_size,))
        Batch_Graphs_list=[]  

        for ind in range(0,Batch_size,1):
            species=Batch_species[ind]
            box_size=Batch_box_size[ind]
            L=jnp.array([box_size[0,1]-box_size[0,0],box_size[1,1]-box_size[1,0]])
            R=Batch_R[ind]
            
            ########################
            """
            R: Node Positions
            L: Box length in x and y
            species: Node type info 0 and 1 """
            
            #1: Calculate pair distances
            R_pair=pair_dist_fn(R,R,L)
            dr2_pair=jnp.sum(jnp.square(R_pair),axis=2)
            dr2_pair=dr2_pair.at[jnp.diag_indices_from(dr2_pair)].set(1e-3)
            dr_pair=jnp.sqrt(dr2_pair)
            
            #3: Creating neigh_list and senders and receivers
            n_list=(dr_pair<cutoffs).astype(int)
            n_list=n_list.at[jnp.diag_indices_from(n_list)].set(0)
            (senders,receivers)=jnp.where(n_list==1)

            #4: Calculating Energy of each pair of atoms
            E_pair=self.lj(dr_pair,sigma,epsilon)
            E_pair=E_pair.at[jnp.diag_indices_from(E_pair)].set(0.0)

            #5: Node features
            node_pe=jnp.sum(E_pair,0).reshape((-1,1))
            #broadcasting node_pe to n dim
            pair_node_pe=jnp.concatenate([node_pe]*N,axis=1).T
            neigh_pe_dist=pair_node_pe*n_list
            deg_node=jnp.sum(n_list,axis=1).reshape((-1,1))
            neigh_pe_sum=jnp.sum(neigh_pe_dist,axis=1).reshape((-1,1))/N
            neigh_pe_mean=neigh_pe_sum/deg_node
            #Node_feats=jnp.concatenate([species.reshape((-1,1)),node_pe,neigh_pe_sum,neigh_pe_mean],axis=1)
            Node_feats=jnp.concatenate([node_pe,neigh_pe_sum,neigh_pe_mean],axis=1)

            #6: Edge Features
            dist_2d=R_pair[senders,receivers,:]
            l_sigma=(dr_pair-sigma)[senders,receivers].reshape((-1,1))
            Edge_feats=jnp.concatenate([dist_2d,l_sigma],axis=1)
            G=jraph.GraphsTuple(nodes=Node_feats,
                                        edges=Edge_feats,
                                        senders=senders,
                                        receivers=receivers,
                                        n_node=jnp.array([Node_feats.shape[0]]), 
                                        n_edge=jnp.array([Edge_feats.shape[0]]),globals=None)

            Energy=jnp.sum(node_pe)/2

            ########################
            
            
            #G,Energy=self.create_G(R,L,species)
            Batch_pe=Batch_pe.at[ind].set(Energy)
            Batch_Graphs_list+=[G] 
            #Node_features+=[np.array(Neigh_Edge_pe_feats)]

        return MDTuple(N=Batch_N,box_size=Batch_box_size,pe=Batch_pe,species=Batch_species,R=Batch_R,neigh_list=Batch_neigh_list, Graph=jraph.batch(Batch_Graphs_list)), Batch_pe-Batch_PE_initial
