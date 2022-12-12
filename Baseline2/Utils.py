
import jax

#Utilities
def matrix_index_fn(matrix,type_a,type_b):
    return matrix[type_a,type_b]


matrix_broadcast_fn=jax.jit(jax.vmap(jax.vmap(matrix_index_fn,(None,0,None)),(None,None,0)))
#To broadscast pair potential parameters to shape (N,N) from (N_species,N_species)
#Example usage pair_sigma=matrix_broadcast_fn(sigma,species,species) where species is (N,) array of species of each atom


def write_xyz(Filepath,Name,R):
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
        #f.write(Name)
        f.write("\nLattice=\""+str(box_size)+"0.0 0.0 0.0 "+str(box_size)+" 0.0 0.0 0.0 "+str(box_size)+"\" Properties=species:S:1:pos:R:"+str(spatial_dim)+":local_energy:R:1 Time="+str(frame_i))
        for i in range(R.shape[0]):
            f.write("\n"+str(species[i])+"\t"+str(Traj[frame_i,i,0])+"\t"+str(Traj[frame_i,i,1])+"\t"+str(Traj[frame_i,i,2])+"\t"+str(Node_energies[frame_i,i]))
