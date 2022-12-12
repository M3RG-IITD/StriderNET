
from Utils import *
import numpy as onp
import scipy.stats as st
from jax.config import config  
config.update('jax_enable_x64', True)

import jax.numpy as jnp
from jax import random,jit,lax,vmap
import jax
from jax import ops
from jax.example_libraries import optimizers
from jax.lax import fori_loop
import jraph
import time

from jax_md import space, smap, energy, minimize, quantity, simulate,partition, nn, interpolate, util
from typing import Any, NamedTuple,  Optional, Union
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from jax._src import prng
Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]
import timeit
import matplotlib
import matplotlib.pyplot as plt



#Stillinger weber system

from functools import wraps, partial
from typing import Callable, Tuple, TextIO, Dict, Any, Optional
from jax.tree_util import tree_map


maybe_downcast = util.maybe_downcast

# Types


f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat

# Stillinger-Weber Potential


def _sw_angle_interaction(gamma: float, sigma: float, cutoff: float,
                          dR12: Array, dR13: Array) -> Array:
    """The angular interaction for the Stillinger-Weber potential.
  This function is defined only for interaction with a pair of
  neighbors. We then vmap this function three times below to make
  it work on the whole system of atoms.
  Args:
    gamma: A scalar used to fit the angle interaction.
    sigma: A scalar that sets the distance scale between neighbors.
    cutoff: The cutoff beyond which the interactions are not
      considered. The default value should not be changed for the
      default SW potential.
    dR12: A d-dimensional vector that specifies the displacement
      of the first neighbor. This potential is usually used in three
      dimensions.
    dR13: A d-dimensional vector that specifies the displacement
      of the second neighbor.

  Returns:
    Angular interaction energy for one pair of neighbors.
  """
    a = cutoff / sigma
    dr12 = space.distance(dR12)
    dr13 = space.distance(dR13)
    dr12 = jnp.where(dr12 < cutoff, dr12, 0)
    dr13 = jnp.where(dr13 < cutoff, dr13, 0)
    term1 = jnp.exp(gamma / (dr12 / sigma - a) + gamma / (dr13 / sigma - a))
    cos_angle = quantity.cosine_angle_between_two_vectors(dR12, dR13)
    term2 = (cos_angle + 1./3)**2
    within_cutoff = (dr12>0) & (dr13>0) & (jnp.linalg.norm(dR12-dR13)>1e-5)
    return jnp.where(within_cutoff, term1 * term2, 0)
    sw_three_body_term = vmap(vmap(vmap(
    _sw_angle_interaction, (0, None)), (None, 0)), 0)
    
def _sw_radial_interaction(sigma: float, B: float, cutoff: float, r: Array
                           ) -> Array:
    """The two body term of the Stillinger-Weber potential."""
    a = cutoff / sigma
    p = 4
    term1 = (B * (r / sigma)**(-p) - 1.0)
    within_cutoff = (r > 0) & (r < cutoff)
    r = jnp.where(within_cutoff, r, 0)
    term2 = jnp.exp(1 / (r / sigma - a))
    return jnp.where(within_cutoff, term1 * term2, 0.0)


def stillinger_weber(displacement: DisplacementFn,
                     sigma: float = 2.0951,
                     A: float = 7.049556277,
                     B: float = 0.6022245584,
                     lam: float = 21.0,
                     gamma: float = 1.2,
                     epsilon: float = 2.16826,
                     three_body_strength: float =1.0,
                     cutoff: float = 3.77118) -> Callable[[Array], Array]:
    """.. _sw-pot:

  Computes the Stillinger-Weber potential.

  The Stillinger-Weber (SW) potential [#stillinger]_ which is commonly used to
  model silicon and similar systems. This function uses the default SW
  parameters from the original paper. The SW potential was originally proposed
  to model diamond in the diamond crystal phase and the liquid phase, and is
  known to give unphysical amorphous configurations [#holender]_ [#barkema]_ .
  For this reason, we provide a `three_body_strength` parameter. Changing this
  number to `1.5` or `2.0` has been know to produce more physical amorphous
  phase, preventing most atoms from having more than four nearest neighbors.
  Note that this function currently assumes nearest-image-convention.

  Args:
    displacement: The displacement function for the space.
    sigma: A scalar that sets the distance scale between neighbors.
    A: A scalar that determines the scale of two-body term.
    B: A scalar that determines the scale of the :math:`1 / r^p` term.
    lam: A scalar that determines the scale of the three-body term.
    epsilon: A scalar that sets the total energy scale.
    gamma: A scalar used to fit the angle interaction.
    three_body_strength:
      A scalar that determines the relative strength
      of the angular interaction. Default value is `1.0`, which works well
      for the diamond crystal and liquid phases. `1.5` and `2.0` have been used
      to model amorphous silicon.
  Returns:
    A function that computes the total energy.

  .. rubric:: References
  .. [#stillinger] Stillinger, Frank H., and Thomas A. Weber. "Computer
    simulation of local order in condensed phases of silicon."
    Physical review B 31.8 (1985): 5262.
  .. [#holender] Holender, J. M., and G. J. Morgan. "Generation of a large
    structure (105 atoms) of amorphous Si using molecular dynamics." Journal of
    Physics: Condensed Matter 3.38 (1991): 7241.
  .. [#barkema] Barkema, G. T., and Normand Mousseau. "Event-based relaxation of
    continuous disordered systems." Physical review letters 77.21 (1996): 4358.
  """
    two_body_fn = partial(_sw_radial_interaction, sigma, B, cutoff)
    three_body_fn = partial(_sw_angle_interaction, gamma, sigma, cutoff)
    three_body_fn = vmap(vmap(vmap(three_body_fn, (0, None)), (None, 0)))

    def node_energy_fn(R, **kwargs):
        d = partial(displacement, **kwargs)
        dR = space.map_product(d)(R, R)
        dr = space.distance(dR)
        first_term = util.high_precision_sum(two_body_fn(dr),axis=1) / 2.0 * A
        second_term = lam *  util.high_precision_sum(util.high_precision_sum(three_body_fn(dR, dR),axis=2),axis=1) / 2.0
        return epsilon * (first_term + three_body_strength * second_term)
    
    def compute_fn(R, **kwargs):
        d = partial(displacement, **kwargs)
        dR = space.map_product(d)(R, R)
        dr = space.distance(dR)
        first_term = util.high_precision_sum(two_body_fn(dr)) / 2.0 * A
        second_term = lam *  util.high_precision_sum(three_body_fn(dR, dR)) / 2.0
        return epsilon * (first_term + three_body_strength * second_term)
    return compute_fn, node_energy_fn


#1: Stillinger Weber Silicon system, with random initialization
def SW_Silicon(key :KeyArray,
              N :int =216,
              spatial_dimension :int =3
                ):
    
    #1 : Create box
    box_size=quantity.box_size_at_number_density(particle_count=N,number_density=0.4832,spatial_dimension=spatial_dimension)
    displacement_fn, shift_fn = space.periodic(box_size)
    #3 : Create composition
    N_species=jnp.array([1])
    species=jnp.array([0]*N)
    
    
    #4 : Create random configuration
    R = box_size * random.uniform(key, (N, spatial_dimension), dtype=jnp.float64)
    
    #5 : Pair_and disp functions
    Disp_Vec_fn= jax.jit(space.map_product(displacement_fn))
    def pair_dist_fn(R):
        dR = Disp_Vec_fn(R, R)
        dr = space.distance(dR)
        return dr
    pair_dist_fn=jax.jit(pair_dist_fn)

    #6 : Get energy and disp_fn
    Total_energy_fn,Node_energy_fn=stillinger_weber(displacement_fn, sigma=2.0951, A=7.049556277, B=0.6022245584, lam=21.0, gamma=1.2, epsilon=2.16826, three_body_strength=1.0, cutoff=3.77118)
    Total_energy_fn=jax.jit(Total_energy_fn)
    Node_energy_fn=jax.jit(Node_energy_fn)
    return N_species,box_size, species,R,Disp_Vec_fn,pair_dist_fn,Node_energy_fn,Total_energy_fn,displacement_fn, shift_fn 

#2: CSH sytem
def CSH(key :KeyArray,
        N: int=400, 
        spatial_dimension :int=3,
        cutoff=150
        ):
    
    #1: Define pair potential
    def CSH_mie(r):
        """
        C=4
        -Calculatrs mie potential for CSH
        Args:
        r: radial distance between pairs of atoms (1,)
        
        Returns:
        CSH potential, same as r.shape
        """
        sigma=50
        eps=2660.7742553
        cutoff=150
        within_cutoff = (r > 0) & (r < cutoff)
        a=jnp.power((sigma/r),14)
        return jnp.where(within_cutoff, 4*eps*a*jnp.add(a,-1),0.0)
    
    #2 : Create box
    box_size=quantity.box_size_at_number_density(particle_count=N,number_density=7.55e-6,spatial_dimension=spatial_dimension)
    displacement_fn, shift_fn = space.periodic(box_size)
    #3 : Create composition
    N_species=jnp.array([1])
    species=jnp.array([0]*N)
    
    #4 : Create random configuration
    R = box_size * random.uniform(key, (N, spatial_dimension), dtype=jnp.float64)
    #Calculating pair distances
    Disp_Vec_fn= jax.jit(space.map_product(displacement_fn))
    def pair_dist_fn(R):
        dR = Disp_Vec_fn(R, R)
        dr = space.distance(dR)
        return dr
    pair_dist_fn=jax.jit(pair_dist_fn)
    
    #6: Calculate pair energies, node energies
    def pair_energies_fn(R):
        dr_pair = pair_dist_fn(R)
        E_pair=CSH_mie(dr_pair)
        return E_pair
    pair_energies_fn=jax.jit(pair_energies_fn)

    def Node_energy_fn(R):
        return jnp.sum(pair_energies_fn(R),axis=1)

    Node_energy_fn=jax.jit(Node_energy_fn)
    def Total_energy_fn(R):
        return jnp.sum(Node_energy_fn(R))*0.5
    
    Total_energy_fn=jax.jit(Total_energy_fn)

    return N_species,box_size, species,R,Disp_Vec_fn,pair_dist_fn,Node_energy_fn,Total_energy_fn,displacement_fn, shift_fn 

    
#3: Lennard Jones sytem
def lennard_jones(key :KeyArray,
                N: int=38, 
                spatial_dimension :int=3,
                composition :jnp.ndarray=jnp.array([80,20]),
                sigma:jnp.ndarray =jnp.array([[1.0 ,0.8],[0.8 ,0.88]],dtype=jnp.float32),
                epsilon:jnp.ndarray =jnp.array([[1.0 ,1.5],[1.5 ,0.5]],dtype=jnp.float32),
                cutoffs:jnp.ndarray =jnp.array([[1.5 ,1.25],[1.25 ,2.0]],dtype=jnp.float32)
            ):

    #1: Define pair potential
    def lj(r,sigma,eps,cutoff):
        """
        -Calculatrs lennard jones potential energy
        Args:
        r: radial distance between pairs of atoms (1,)
        sigma: sigma parameter (1,)
        epsilon: epsilon parameter (1,)
        cutoff : zero energy beyond cutoff
        Returns:
        LJ potential (1,)
        """
        within_cutoff = (r > 0) & (r < cutoff)
        a=jnp.power((sigma/r),6)
        return jnp.where(within_cutoff, 4*eps*a*jnp.add(a,-1),0.0)
    
    
    
    #2 : Create box
    box_size=quantity.box_size_at_number_density(particle_count=N,number_density=1.2,spatial_dimension=spatial_dimension)
    displacement_fn, shift_fn = space.periodic(box_size)
    #3 : Create composition
    N_species=composition.shape[0]
    N_composition=(jnp.round(N*composition/jnp.sum(composition))).astype('int32')
    #species=jnp.concatenate([jnp.array([k]*N_composition[k]) for k in range(N_composition.shape[0])])
    species=jnp.array(onp.load("species.npy"))
    #4 : Create random configuration
    R = box_size * random.uniform(key, (N, spatial_dimension), dtype=jnp.float64)
    
    #Calculating pair distances
    Disp_Vec_fn= jax.jit(space.map_product(displacement_fn))
    def pair_dist_fn(R):
        dR = Disp_Vec_fn(R, R)
        dr = space.distance(dR)
        return dr
    pair_dist_fn=jax.jit(pair_dist_fn)
  
    #5: Creating neigh_list and senders and receivers
    
    pair_cutoffs=matrix_broadcast_fn(cutoffs,species,species)
    pair_sigma=matrix_broadcast_fn(sigma,species,species)
    pair_epsilon=matrix_broadcast_fn(epsilon,species,species)
    
    #6: Calculate pair energies, node energies
    def pair_energies_fn(R):
        pair_cutoffs=matrix_broadcast_fn(cutoffs,species,species)
        pair_sigma=matrix_broadcast_fn(sigma,species,species)
        pair_epsilon=matrix_broadcast_fn(epsilon,species,species)
        dr_pair = pair_dist_fn(R)
        E_pair=lj(dr_pair,pair_sigma,pair_epsilon,pair_cutoffs)
        return E_pair
    pair_energies_fn=jax.jit(pair_energies_fn)

    def Node_energy_fn(R):
        return jnp.sum(pair_energies_fn(R),axis=1)
    Node_energy_fn=jax.jit(Node_energy_fn)

    def Total_energy_fn(R):
        return jnp.sum(Node_energy_fn(R))*0.5
    Total_energy_fn=jax.jit(Total_energy_fn)

    return N_species,box_size, species,R,Disp_Vec_fn,pair_dist_fn,Node_energy_fn,Total_energy_fn,displacement_fn, shift_fn 

    
