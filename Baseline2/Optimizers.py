import numpy as onp
import scipy.stats as st
from jax.config import config  
config.update('jax_enable_x64', True)

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

from jax._src import prng
Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]
import timeit

#MINIMIZERS
#Gradient descent minimization
def SimpleGD(lr,Nstep,Init_state,Total_energy_fn,shift_fn):
    energy_fn = Total_energy_fn
    init,apply=minimize.gradient_descent(energy_fn,shift_fn,lr)
    apply=jit(apply)
    Traj=jnp.zeros((Nstep,*Init_state.shape))
    state=init(Init_state)
    def step_fn(i,packed_state_Traj):
        state,Traj=packed_state_Traj[0],packed_state_Traj[1]
        new_state=apply(state)
        Traj=Traj.at[i].set(new_state)
        return new_state, Traj
    return jax.lax.fori_loop(0,Nstep,jax.jit(step_fn),(state,Traj))


#FIRE minimization
def FIRE_desc(dt_start,Nstep,Init_state,Total_energy_fn,shift_fn):
    """
    Performs Fast Inertial Relaxation Engine Minimization 
    Returns Fire_State, Use .position to get configuration

    """
    energy_fn = Total_energy_fn
    init,apply=minimize.fire_descent(energy_fn,
                 shift_fn,
                 dt_start=dt_start,
                 dt_max=4*dt_start,
                 n_min=5,
                 f_inc=1.1,
                 f_dec=0.5,
                 alpha_start=0.1,
                 f_alpha=0.99)
    Traj=jnp.zeros((Nstep,*Init_state.shape))
    state=init(Init_state)
    def step_fn(i,packed_state_Traj):
        state,Traj=packed_state_Traj[0],packed_state_Traj[1]
        new_state=apply(state)
        Traj=Traj.at[i].set(new_state.position)
        return new_state, Traj
    return jax.lax.fori_loop(0,Nstep,jax.jit(step_fn),(state,Traj))


#Liquid Initialiser
def NVT_Liq_Init(key,Temp,dt,Nstep,write_every,Init_state,Total_energy_fn,shift_fn):
    #Thermalizing at liquid temperature of T=Temp
    def energy_fn(R,**kwargs):
        return Total_energy_fn(R)
    energy_fn=jax.jit(energy_fn)
    init, apply = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT=Temp)
    state = init(key, Init_state,mass=1.0)
    #kT = lambda t: np.where(t < 5000.0 * dt, 0.1, 0.01)
    kT = lambda t: Temp         #COnst Temp
    
    def step_fn(i, state_and_log):
        state, log = state_and_log

        t = i * dt

        # Log information about the simulation.
        T = quantity.temperature(momentum=state.momentum)
        log['kT'] = log['kT'].at[i].set(T)
        H = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT(t))
        log['H'] = log['H'].at[i].set(H)
        # Record positions every `write_every`steps.
        log['position'] = lax.cond(i % write_every == 0,
                                    lambda p: \
                                    p.at[i // write_every].set(state.position),
                                    lambda p: p,
                                    log['position'])
        # Take a simulation step.
        state = apply(state, kT=kT(t))
        return state, log
    log = {
            'kT':jnp.zeros((Nstep,)),
            'H': jnp.zeros((Nstep,)),
            'PE': jnp.zeros((Nstep,)),
            'KE': jnp.zeros((Nstep,)),
            'position': jnp.zeros((Nstep // write_every,) + Init_state.shape) 
        }
    
    state, log = lax.fori_loop(0, Nstep, step_fn, (state, log)) 
    return state.position, log
