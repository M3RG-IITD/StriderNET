import numpy as onp
import scipy.stats as st
from jax.config import config  
config.update('jax_enable_x64', True)

import jax.numpy as jnp
from jax import random,jit,lax,ops
import jax
from jax.example_libraries import optimizers
from jax.lax import fori_loop
import jraph

import jax_md
from jax_md import space, smap, energy, minimize, quantity, simulate
from typing import Any, NamedTuple,  Optional, Union
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from jax._src import prng
import timeit

import Systems
from Systems import lennard_jones, SW_Silicon, CSH
from Optimizers import *
from Utils import *

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]


etas=jnp.array([0.001, 0.01 ,0.05, 0.1, 0.2, 0.4,1,2])
radial_feats_fn=radial_symmetry_functions(etas,2.5,1)        #N_species=1 considered here 
radial_feats_fn=jax.jit(jax.vmap(radial_feats_fn,(0,0)))
          

class MDTuple(NamedTuple):
    N          : jnp.ndarray    #No. of particles (B_sz,)
    N_types    : jnp.ndarray    #No. of particles types (B_sz,)
    box_size   : jnp.ndarray    #Size of box [(B_sz,spatial_dim,2) array of [[xlo, xhi],[ylo,yhi],...,[zlo,zhi]]
    pe         : jnp.ndarray    #potential energy of system (B_sz,)
    species    : jnp.ndarray    #Atom types list (B_sz,N) from the set of types={0,1,2,...,N_types-1}
    R          : jnp.ndarray    #Position vectors of particles (B_sz,N, spatial_dim)
    neigh_list : jnp.ndarray    #neigh adjacency matrix  (B_sz,N,N)
    Graph      : Optional[jraph.GraphsTuple]   #Graph structure

class Generic_system:
    def create_batched_States(self,key :KeyError,System='LJ',spatial_dimension: int=3,N :int=38, N_sample :int=100,Batch_size: int=10,Traj=None,isTraj=False):
        """
        Available Systems = ['LJ', 'CSH', 'SW_Si']
        -Creates Train, Val, and Test data of batched system_states along with batched graphs and random shuffling
        -Current split is 60:20:20
        """
        Available_Systems = ['LJ', 'CSH', 'SW_Si']
        System_fns={'LJ':Systems.lennard_jones, 'CSH':Systems.CSH, 'SW_Si':Systems.SW_Silicon}
        System_Liq_Temps={'LJ':2.0, 'CSH':1000, 'SW_Si':2000}
        if(System not in Available_Systems):
            raise  ValueError("Available systems are only "+str(Available_Systems))
        
        Chosen_sys_fn=System_fns[System]
        key1, key2 = jax.random.split(key,2)
    
        #1 Create a random sysytem configuration:
        N_species,box_size,species,R,Disp_Vec_fn,pair_dist_fn,Node_energy_fn,Total_energy_fn,displacement_fn, shift_fn =Chosen_sys_fn(key1,N=N,spatial_dimension=spatial_dimension)
        Box_lengths=jnp.array([box_size]*spatial_dimension)
        
        #2:Cutoffs and sigma parameters for different systems
        if(System=='LJ'):
            species=jnp.array(onp.load("species.npy"))
            sigma:jnp.ndarray =jnp.array([[1.0 ,0.8],[0.8 ,0.88]],dtype=jnp.float32)
            pair_sigma=matrix_broadcast_fn(sigma,species,species)
            cutoffs:jnp.ndarray =jnp.array([[1.5 ,1.25],[1.25 ,2.0]],dtype=jnp.float32)
            pair_cutoffs=matrix_broadcast_fn(cutoffs,species,species)
        
        elif(System=='CSH'):
            cutoff=75  #For graph creation
            sigma=50   #Potential's constant
            sigma:jnp.ndarray =jnp.array([[sigma,sigma],[sigma,sigma]],dtype=jnp.float32)
            pair_sigma=matrix_broadcast_fn(sigma,species,species)
            cutoffs:jnp.ndarray =jnp.array([[cutoff,cutoff],[cutoff,cutoff]],dtype=jnp.float32)
            pair_cutoffs=matrix_broadcast_fn(cutoffs,species,species)
        elif(System=='SW_Silicon'):
            cutoff=3.77118
            sigma=2.0951
            sigma:jnp.ndarray =jnp.array([sigma],dtype=jnp.float32)
            pair_sigma=matrix_broadcast_fn(sigma,species,species)
            cutoffs:jnp.ndarray =jnp.array([cutoff],dtype=jnp.float32)
            pair_cutoffs=matrix_broadcast_fn(cutoffs,species,species)
        else:
            raise  ValueError("Available systems are "+str(Available_Systems))
        
        Batch_Disp_Vec_fn=jax.jit(jax.vmap(Disp_Vec_fn,(0,0)))
        Batch_pair_dist_fn=jax.jit(jax.vmap(pair_dist_fn,0))
        Batch_Node_energy_fn=jax.jit(jax.vmap(Node_energy_fn,0))
        Batch_Total_energy_fn=jax.jit(jax.vmap(Total_energy_fn,0))
        Batch_pair_cutoffs=jnp.stack([pair_cutoffs]*Batch_size)
        Batch_pair_sigma=jnp.stack([pair_sigma]*Batch_size)
        Batch_species=jnp.stack([species]*Batch_size)
        Batch_box_size=jnp.stack([jnp.array([[0, box_size]]*spatial_dimension)]*Batch_size)
        Batch_N=jnp.ones((Batch_size,))*N
            
        #3:Initialising minimization
        Min_R=SimpleGD(1e-6,1000,R,Total_energy_fn,shift_fn)[0]

        #4:NVT_Liquid Initialisation
        key3, key4 = jax.random.split(key1,2)
            
        if(System=='LJ'):
            if(Traj==None):
                Traj=jnp.array(onp.load("DatasetLJ100_3D_Pos.npy"))[:N_sample,:,:]
            print(Batch_Total_energy_fn(Traj))
        elif(System=='CSH'):
            if(isTraj==False):
                #Perform NVT liquid initialisation
                _,log=NVT_Liq_Init(key4,System_Liq_Temps[System],1e-2,N_sample*10,10,Min_R,Total_energy_fn,shift_fn)
                Traj=log['position']
                onp.save("CSH_Starting_Traj",Traj) 
            print(Batch_Total_energy_fn(Traj))
        else:
            raise NotImplementedError("Available systems are "+str(Available_Systems))
        
        #5:Creating dataset from initialised trajectory
        N_sample=Traj.shape[0]
        key5, key6 = jax.random.split(key3,2)
        Shuffled_index=jax.random.permutation(key3, onp.arange(0,N_sample,1))  
        G_indeces=jax.random.permutation(key5, onp.arange(0,N_sample-Batch_size+1,Batch_size))    
        
        Systates=[]
        for k in G_indeces:
            #Shuffle indeces within batch
            key6, key = jax.random.split(key6,2)
            Batch_indeces=jax.random.permutation(key,Shuffled_index[k:k+Batch_size:1])        
            
            Batch_R=Traj[Batch_indeces,:,:]
            Batch_G, Batch_pe, Batch_neigh_list=self.create_G_batched(Batch_R,Batch_species,Batch_pair_cutoffs,Batch_pair_sigma,Batch_Disp_Vec_fn,Batch_Node_energy_fn,Batch_Total_energy_fn)
            Systates+=[MDTuple(N=Batch_N,N_types=N_species,box_size=Batch_box_size,pe=Batch_pe,species=Batch_species,R=Batch_R,neigh_list=Batch_neigh_list, Graph=Batch_G)]
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
        
        return Train_Batch,Val_Batch,Test_Batch, shift_fn,Batch_pair_cutoffs,Batch_pair_sigma,Batch_Disp_Vec_fn,Batch_Node_energy_fn,Batch_Total_energy_fn,displacement_fn, shift_fn ,Disp_Vec_fn
    
    #Create_Batched_graph_function
    def create_G_batched(self,Batch_R,Batch_species,Batch_pair_cutoffs,Batch_pair_sigma,Batch_Disp_Vec_fn,Batch_Node_energy_fn,Batch_Total_energy_fn):
        """
        Batch_R: Node Positions (B_sz,N,spatial_dim)
        Disp_vec_fn: Calculates distance between atoms considering periodic boundaries
        species: Node type info 0 and 1
        cutoffs: pair cutoffs (N,N) shape
        sigma  : pair sigma   (N,N)
        
        Returns
        Batch_G      :batched graph
        Batch_Energy :PE of each graph
        Batch_n_list :neighbor adjacency matrix of each graph
         """
        B_sz=Batch_R.shape[0]
        N=Batch_R.shape[1]
        #1: Calculate pair distances
        Batch_R_pair = Batch_Disp_Vec_fn(Batch_R, Batch_R)
        Batch_dr_pair =jax_md.space.distance(Batch_R_pair)
            
        #Radial symmetry 
        #Batch_radial_feats=radial_feats_fn(Batch_dr_pair,jnp.zeros(Batch_species.shape))
        
        #3: Creating neigh_list and senders and receivers
        Batch_n_list=((Batch_dr_pair<Batch_pair_cutoffs) & (Batch_dr_pair>0)).astype(int)
        Batch_node_pe=Batch_Node_energy_fn(Batch_R)
        
        Batch_Graphs=[]
        Batch_pe=[]
        for Batch_i in range(B_sz):
            #5: Node features
            node_pe=Batch_node_pe[Batch_i].reshape(-1,1)
            n_list=Batch_n_list[Batch_i]
            R_pair=Batch_R_pair[Batch_i]
            dr_pair=Batch_dr_pair[Batch_i]
            pair_sigma=Batch_pair_sigma[Batch_i]
            #broadcasting node_pe to N dim
            (senders,receivers)=jnp.where(n_list==1)
            pair_node_pe=jnp.concatenate([node_pe]*N,axis=1).T
            neigh_pe_dist=pair_node_pe*n_list
            deg_node=jnp.sum(n_list,axis=1).reshape((-1,1))
            neigh_pe_sum=jnp.sum(neigh_pe_dist,axis=1).reshape((-1,1))/N
            neigh_pe_mean=neigh_pe_sum/deg_node
            
            
        
            #Node_feats=jnp.concatenate([Batch_species[Batch_i].reshape((-1,1)),node_pe,neigh_pe_sum,neigh_pe_mean,Batch_radial_feats[Batch_i]],axis=1)
            #Node_feats=jnp.concatenate([Batch_species[Batch_i].reshape((-1,1)),node_pe,neigh_pe_sum,neigh_pe_mean],axis=1)
            Node_feats=jnp.concatenate([node_pe,neigh_pe_sum,neigh_pe_mean],axis=1)

            #6: Edge Features
            dist_2d=R_pair[senders,receivers,:]
            l_sigma=(dr_pair-pair_sigma)[senders,receivers].reshape((-1,1))
            Edge_feats=jnp.concatenate([dist_2d,l_sigma],axis=1)
            G=jraph.GraphsTuple(nodes=Node_feats,
                                        edges=Edge_feats,
                                        senders=senders,
                                        receivers=receivers,
                                        n_node=jnp.array([Node_feats.shape[0]]), 
                                        n_edge=jnp.array([Edge_feats.shape[0]]),globals=None)
            
            Batch_Graphs+=[G]
            Batch_pe+=[jnp.sum(node_pe)/2]
        Batch_G=jraph.batch(Batch_Graphs)
        Batch_Energy=jnp.array(Batch_pe)
        return Batch_G, Batch_Energy, Batch_n_list
    
    
    #Create_graph_function
    def create_G(self,R,species,pair_cutoffs,pair_sigma,Disp_Vec_fn,Node_energy_fn,Total_energy_fn):
        """
        R: Node Positions
        Disp_vec_fn: Calculates distance between atoms considering periodic boundaries
        species: Node type info 0 and 1
        cutoffs: pair cutoffs (N,N) shape
        sigma  : pair sigma   (N,N)
         """
        #1: Calculate pair distances
        N=R.shape[0]
        R_pair = Disp_Vec_fn(R, R)
        dr_pair =jax_md.space.distance(R_pair)
        
        #3: Creating neigh_list and senders and receivers
        
        n_list=(dr_pair<pair_cutoffs).astype(int)
        n_list=n_list.at[jnp.diag_indices_from(n_list)].set(0)
        (senders,receivers)=jnp.where(n_list==1)

        #5: Node features
        node_pe=Node_energy_fn(R)
        pair_node_pe=jnp.concatenate([node_pe]*N,axis=1).T  #broadcasting node_pe to N dim
        neigh_pe_dist=pair_node_pe*n_list
        deg_node=jnp.sum(n_list,axis=1).reshape((-1,1))
        neigh_pe_sum=jnp.sum(neigh_pe_dist,axis=1).reshape((-1,1))/N
        neigh_pe_mean=neigh_pe_sum/deg_node
            
        Node_feats=jnp.concatenate([species.reshape((-1,1)),node_pe,neigh_pe_sum,neigh_pe_mean],axis=1)
        #Node_feats=jnp.concatenate([node_pe,neigh_pe_sum,neigh_pe_mean],axis=1)

        #6: Edge Features
        dist_3d=R_pair[senders,receivers,:]
        l_sigma=(dr_pair-pair_sigma)[senders,receivers].reshape((-1,1))                               
        Edge_feats=jnp.concatenate([dist_3d,l_sigma],axis=1)
        G=jraph.GraphsTuple(nodes=Node_feats,
                                    edges=Edge_feats,
                                    senders=senders,
                                    receivers=receivers,
                                    n_node=jnp.array([Node_feats.shape[0]]), 
                                    n_edge=jnp.array([Edge_feats.shape[0]]),globals=None)
        
        Energy=jnp.sum(node_pe)/2
        return G, Energy, n_list
    

    def finalize_plot(self,shape=(1, 1)):
        """Helper function for size of plots"""
        plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
        plt.tight_layout()
    
    def plot_batched(self,Systate: MDTuple,epoch_id=0,batch_id=0,step_id=0,node_ids=None,Edges=True,save=False):
        """To plot batched system state"""
        ms=65
        N,N_types,box_size,pe,species,R,n_adj,G=Systate
        G_list=jraph.unbatch(G)  #Unbatch graphs to list of graphs
        for k in range(len(G_list)):
            fig,ax=plt.subplots(1,1,figsize=(10,8))
            G=G_list[k]
            R_plt=R[k]
            colors=G.nodes[:,1]
            senders=G.senders
            receivers=G.receivers
            v_min=onp.round(onp.mean(colors)-2*onp.std(colors),decimals=1)
            v_max=onp.round(onp.mean(colors)+2*onp.std(colors),decimals=1)
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
        N,N_types,box_size,pe,species,R,n_adj,G=Systate
        senders=G.senders
        R_plt = R
        colors=[]
        if(color_pe==True):
            colors=G.nodes[:,2]
            v_min=onp.round(onp.mean(colors)-2*onp.std(colors),decimals=1)
            v_max=onp.round(onp.mean(colors)+2*onp.std(colors),decimals=1)
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
   
        
    def multi_disp_node(self,Disp_Vecs,Systate :MDTuple,shift_fn,Batch_pair_cutoffs,Batch_pair_sigma,Batch_Disp_Vec_fn,Batch_Node_energy_fn,Batch_Total_energy_fn):
        """
        -To displace several nodes at once and return updated batched system state and change in PE
        -Disp_Vecs   : Displacement vectors of nodes to displace   Shape : (B_sz,k,space_dim)
        -Systate     : Current system state
        """
        Batch_N,Batch_N_species,Batch_box_size,Batch_PE_initial,Batch_species,Batch_R,Batch_neigh_list,Batch_OldG=Systate
        Batch_size=len(Batch_N)
        New_Batch_R=shift_fn(Batch_R,Disp_Vecs)
        Batch_G, Batch_pe, Batch_neigh_list=self.create_G_batched(New_Batch_R,Batch_species,Batch_pair_cutoffs,Batch_pair_sigma,Batch_Disp_Vec_fn,Batch_Node_energy_fn,Batch_Total_energy_fn)
        return MDTuple(N=Batch_N,N_types=Batch_N_species,box_size=Batch_box_size,pe=Batch_pe,species=Batch_species,R=New_Batch_R,neigh_list=Batch_neigh_list, Graph=Batch_G), Batch_pe-Batch_PE_initial

        