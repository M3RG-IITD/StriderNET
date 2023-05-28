## StriderNet: A Graph Reinforcement Learning Approach to Optimize Atomic Structures on Rough Energy Landscapes
Graph neural network based policy gradient learning to get ultra stable glass structure.

    State  : Glass configuration as graph 

    Action : Displacing the atoms(Continuous)
    
    Reward : Reduction in energy ( -DE)
    
   ![Logo](./src/LJSystem_optimize_schematic.png)
## Authors
	-Vaibhav Bihani, Deparment of Civil Engineering, Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016
	-Sahil Manchanda,Department of Computer Science and Engineering,Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016
	-Sayan Ranu∗,Department of Computer Science and Engineering,Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016 and,Yardi School of Artificial Intelligence, Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016
	-N. M. Anoop Krishnan†, Department of Civil Engineering, Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016 and, Yardi School of Artificial Intelligence, Indian Institute of Technology Delhi, Hauz Khas, New Delhi, India 110016

## Environment settings
    python      > 3.7
    jax         > 0.4.1                 
    jax-md      > 0.2.24                 
    jaxlib      > 0.4.1                 
    flax        > 0.6.3                 
    jraph       > 0.0.6.dev0            
    optax       > 0.1.4                 

**Example Usage**

To run the training of model :
```
python -u -m src.Train.py
```

