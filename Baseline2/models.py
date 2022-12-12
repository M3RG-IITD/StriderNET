# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless r/equired by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library of Graph Neural Network models."""

import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Union, Tuple, Sequence, NamedTuple
from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jraph._src import graph as gn_graph
from jraph._src import utils
from jax._src import prng
from jax import random, vmap
from functools import partial
import jraph

import time


#Debugging
#config.update('jax_disable_jit', True)
#config.update("jax_debug_nans", True)

# As of 04/2020 pytype doesn't support recursive types.
# pytype: disable=not-supported-yet

Array = Any
KeyArray = Union[Array, prng.PRNGKeyArray]
ArrayTree = Union[jnp.ndarray,
                  Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# All features will be an ArrayTree.
NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

# Signature:
# (edges of each node to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToNodesFn = Callable[
    [EdgeFeatures, jnp.ndarray, int], NodeFeatures]

# Signature:
# (nodes of each graph to be aggregated, segment ids, number of segments) ->
# aggregated nodes
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edges of each graph to be aggregated, segment ids, number of segments) ->
# aggregated edges
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, jnp.ndarray, int],
                                     Globals]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# attention weights
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree]

# Signature:
# (edge features, weights) -> edge features for node update
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]

# Signature:
# (edges to be normalized, segment ids, number of segments) ->
# normalized edges
AttentionNormalizeFn = Callable[[EdgeFeatures, jnp.ndarray, int], EdgeFeatures]

# Signature:
# (edge features, sender node features, receiver node features, globals) ->
# updated edge features
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures]

GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]

# Signature:
# (node features, outgoing edge features, incoming edge features,
#  globals) -> updated node features
GNNode_to_P_Fn = Callable[[NodeFeatures], NodeFeatures] #P: Probabilty of selecting node



def Policy_Net(
    node_to_prob_fn:GNNode_to_P_Fn,   #Node features to node probabilities function
    initial_edge_embed_fn: Optional[GNUpdateEdgeFn],
    initial_node_embed_fn: Optional[GNUpdateEdgeFn],
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_mean,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils
    .segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils
    .segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils
    .segment_softmax,
        attention_reduce_fn: Optional[AttentionReduceFn] = None,
        N=1,):
    """Returns a method that applies a configured GraphNetwork.

    This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

    There is one difference. For the nodes update the class aggregates over the
    sender edges and receiver edges separately. This is a bit more general
    than the algorithm described in the paper. The original behaviour can be
    recovered by using only the receiver edge aggregations for the update.

    In addition this implementation supports softmax attention over incoming
    edge features.

    Example usage::

      gn = GraphNetwork(update_edge_function,
      update_node_function, **kwargs)
      # Conduct multiple rounds of message passing with the same parameters:
      for _ in range(num_message_passing_steps):
        graph = gn(graph)

    Args:
      update_edge_fn: function used to update the edges or None to deactivate edge
        updates.
      update_node_fn: function used to update the nodes or None to deactivate node
        updates.
      update_global_fn: function used to update the globals or None to deactivate
        globals updates.
      aggregate_edges_for_nodes_fn: function used to aggregate messages to each
        node.
      aggregate_nodes_for_globals_fn: function used to aggregate the nodes for the
        globals.
      aggregate_edges_for_globals_fn: function used to aggregate the edges for the
        globals.
      attention_logit_fn: function used to calculate the attention weights or
        None to deactivate attention mechanism.
      attention_normalize_fn: function used to normalize raw attention logits or
        None if attention mechanism is not active.
      attention_reduce_fn: function used to apply weights to the edge features or
        None if attention mechanism is not active.

    Returns:
      A method that applies the configured GraphNetwork.
    """
    def not_both_supplied(x, y): return (
        x != y) and ((x is None) or (y is None))
    if not_both_supplied(attention_reduce_fn, attention_logit_fn):
        raise ValueError(('attention_logit_fn and attention_reduce_fn must both be'
                          ' supplied.'))

    def _ApplyGraphNet(graph):
        """Applies a configured GraphNetwork to a graph.

        This implementation follows Algorithm 1 in https://arxiv.org/abs/1806.01261

        There is one difference. For the nodes update the class aggregates over the
        sender edges and receiver edges separately. This is a bit more general
        the algorithm described in the paper. The original behaviour can be
        recovered by using only the receiver edge aggregations for the update.

        In addition this implementation supports softmax attention over incoming
        edge features.

        Many popular Graph Neural Networks can be implemented as special cases of
        GraphNets, for more information please see the paper.

        Args:
          graph: a `GraphsTuple` containing the graph.

        Returns:
          Updated `GraphsTuple`, 'Vij'.

        ##
        NODEfeature  = X, TYPE
        type --fa-> n_embed_i
        eij  --fb-> e_embed_i
        n_embed = n_embed_i
        e_embed = e_embed_i
        #for loop
            n_embed, sum e_embed --fv-> n_embed
            e_embed, n_embed --fe-> e_embed
        e_embed --ff--> vij
        """
        # pylint: disable=g-long-lambda
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
        # Equivalent to jnp.sum(n_node), but jittable
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
        sum_n_edge = senders.shape[0]
        if not tree.tree_all(
                tree.tree_map(lambda n: n.shape[0] == sum_n_node, nodes)):
            raise ValueError(
                'All node arrays in nest must contain the same number of nodes.')

        sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
        received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
        # Here we scatter the global features to the corresponding edges,
        # giving us tensors of shape [num_edges, global_feat].
        global_edge_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_edge, axis=0, total_repeat_length=sum_n_edge), globals_)
        # Here we scatter the global features to the corresponding nodes,
        # giving us tensors of shape [num_nodes, global_feat].
        global_attributes = tree.tree_map(lambda g: jnp.repeat(
            g, n_node, axis=0, total_repeat_length=sum_n_node), globals_)

        if initial_edge_embed_fn: #fa
            edges = initial_edge_embed_fn(edges, sent_attributes, received_attributes,
                                      global_edge_attributes)

        if initial_node_embed_fn: #fb
            nodes = initial_node_embed_fn(nodes, sent_attributes,
                                          received_attributes, global_attributes)
                
        #Now perform message passing for N times
        for pass_i in range(N):
            if attention_logit_fn:
                logits = attention_logit_fn(edges, sent_attributes, received_attributes,
                                            global_edge_attributes)
                tree_calculate_weights = functools.partial(
                    attention_normalize_fn,
                    segment_ids=receivers,
                    num_segments=sum_n_node)
                weights = tree.tree_map(tree_calculate_weights, logits)
                edges = attention_reduce_fn(edges, weights)

            if update_node_fn:
                sent_attributes = tree.tree_map(
                    lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges)
                received_attributes = tree.tree_map(
                    lambda e: aggregate_edges_for_nodes_fn(
                        e, receivers, sum_n_node),
                    edges)
                nodes = update_node_fn(nodes, sent_attributes,
                                       received_attributes, global_attributes)
                
            if update_edge_fn:
                sent_attributes = tree.tree_map(lambda n: n[senders], nodes)
                received_attributes = tree.tree_map(lambda n: n[receivers], nodes)
                edges = update_edge_fn(edges, sent_attributes, received_attributes,
                                       global_edge_attributes)
                
        if update_global_fn:
            n_graph = n_node.shape[0]
            graph_idx = jnp.arange(n_graph)
            # To aggregate nodes and edges from each graph to global features,
            # we first construct tensors that map the node to the corresponding graph.
            # For example, if you have `n_node=[1,2]`, we construct the tensor
            # [0, 1, 1]. We then do the same for edges.
            node_gr_idx = jnp.repeat(
                graph_idx, n_node, axis=0, total_repeat_length=sum_n_node)
            edge_gr_idx = jnp.repeat(
                graph_idx, n_edge, axis=0, total_repeat_length=sum_n_edge)
            # We use the aggregation function to pool the nodes/edges per graph.
            node_attributes = tree.tree_map(
                lambda n: aggregate_nodes_for_globals_fn(
                    n, node_gr_idx, n_graph),
                nodes)
            edge_attribtutes = tree.tree_map(
                lambda e: aggregate_edges_for_globals_fn(
                    e, edge_gr_idx, n_graph),
                edges)
            # These pooled nodes are the inputs to the global update fn.
            globals_ = update_global_fn(
                node_attributes, edge_attribtutes, globals_)
    
        prob_vi = node_to_prob_fn(nodes)

        # pylint: enable=g-long-lambda
        return gn_graph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge), prob_vi

    return _ApplyGraphNet


# Signature:
# edge features -> embedded edge features
EmbedEdgeFn = Callable[[EdgeFeatures], EdgeFeatures]

# Signature:
# node features -> embedded node features
EmbedNodeFn = Callable[[NodeFeatures], NodeFeatures]

# Signature:
# globals features -> embedded globals features
EmbedGlobalFn = Callable[[Globals], Globals]

class Pol_Net(nn.Module):
    edge_emb_size:int
    node_emb_size: int
    fa_layers: int
    fb_layers: int
    fv_layers: int
    fe_layers: int
    MLP1_layers: int #MLP1 has one additional layer for returning scaler probabilities for each node 
    MLP2_layers:int  #MLP2 for displacement
    spatial_dim :int  
    sigma :float     #Scaling parameter for displacement
    train : bool
    
    message_passing_steps: int
    dropout_rate: float = 0 #Not currently used
    deterministic: bool = True #Not currently used
    fa_activation: Callable[[jnp.ndarray], jnp.ndarray] =jax.nn.leaky_relu#jax.nn.hard_tanh #nn.softplus#jax.nn.hard_tanh# nn.relu #nn
    fb_activation: Callable[[jnp.ndarray], jnp.ndarray] =jax.nn.leaky_relu#jax.nn.hard_tanh #jax.nn.softplus #nn.softplus#jax.nn.hard_tanh# nn.relu
    fv_activation: Callable[[jnp.ndarray], jnp.ndarray] =jax.nn.leaky_relu#jax.nn.hard_tanh #jax.nn.softplus  # jax.nn.hard_tanh#nn.relu   # Use relu 
    fe_activation: Callable[[jnp.ndarray], jnp.ndarray] =jax.nn.leaky_relu#jax.nn.hard_tanh #jax.nn.hard_tanh #jax.nn.hard_tanh#nn.relu
    MLP1_activation: Callable[[jnp.ndarray], jnp.ndarray] =jax.nn.leaky_relu#jax.nn.hard_tanh # jax.nn.softplus #relu leads to inf and nan after normalization
    MLP1_normalize_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.softmax
    MLP2_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.leaky_relu#jax.nn.hard_tanh
    #num_species: int=2     #Two types of atoms A and B
    
    def setup(self):
        fa_features=[self.node_emb_size]*self.fa_layers
        fb_features=[self.edge_emb_size]*self.fb_layers
        fv_features=[self.node_emb_size]*self.fv_layers
        fe_features=[self.edge_emb_size]*self.fe_layers
        MLP1_features=[self.node_emb_size]*self.MLP1_layers+[1]
        MLP2_features=[self.node_emb_size]*self.MLP2_layers+[self.spatial_dim] #[x,y]
        
        self.fa_mlp=[nn.BatchNorm(use_running_average=not self.train,
                     momentum=0.9,epsilon=1e-5,dtype=jnp.float32)]+[nn.Dense(feat) for i,feat in enumerate(fa_features)]
        #self.fa_mlp=[nn.Dense(feat) for i,feat in enumerate(fa_features)]
        self.fb_mlp=[nn.Dense(feat) for i,feat in enumerate(fb_features)]
        self.fv_mlp=[nn.Dense(feat) for i,feat in enumerate(fv_features)]
        self.fe_mlp=[nn.Dense(feat) for i,feat in enumerate(fe_features)]
        self.MLP1_mlp=[nn.Dense(feat) for i,feat in enumerate(MLP1_features)]
        self.MLP2_mlp=[nn.Dense(feat) for i,feat in enumerate(MLP2_features)]
        
    
    def __call__(self, graphs: jraph.GraphsTuple):
       
        #fe
        def update_edge_fn(edges, senders, receivers, globals_):
            #print("\n#Update_Edge_fn")
            '''
            print("edges: ",edges.shape,'\n',edges)
            print("senders: ",senders.shape,'\n',senders) #Senders[i]=node feature of sender of edge 'i'
            print("receivers: ",receivers.shape,'\n',receivers)#Similarly  for receivers
            #print("globals_: ",globals_.shape,'\n',globals_)#globals broadcasted to all edges'''
             
            '''We need to implement hij'=tanh( hij +MLP_sigma(hi|*|hj) |*|' MLP_tanh(hi|*|hj)+'3_body_term'  ) 
               where '*' '''
            #Current implementation hij'=tanh( hij +MLP_sigma(hi||hj)
        
            x = jnp.concatenate([edges,senders,receivers],axis=1) #concatenates node feature to each edge
            #print("Senders:Receievers",x[:10])
            for i, lyr in enumerate(self.fe_mlp):
                x = lyr(x)
                if (i != len(self.fe_mlp) - 1):
                    x = self.fe_activation(x)
            #x=jax.nn.hard_tanh(edges+x)
            #print("Updated_edge")
            #print(x[:10])
            
            return x

        #fv
        def update_node_fn(nodes, sent_edges, received_edges, globals_): #fv
            #print("\n#Update_Node_fn")
            '''print("nodes: ",nodes.shape,'\n',nodes)
            print("sent_edges: ",sent_edges.shape,'\n',sent_edges)# sent_edges[i]=agg_sum_of outgoing edges for node 'i'
            print("received_edges: ",received_edges.shape,'\n',received_edges)#Similarly
            #print("globals_: ",globals_.shape,'\n',globals_)#global broadcasted to node'''
            
            '''We need to implement hi'=tanh( hi +MLP_sigma(hi||hij) |*|' MLP_tanh(hi||hij)  ) 
               where '*' '''
            #Current implementation hi'=tanh( hi +MLP_sigma(hi||hij)
            #print(nodes[:10],received_edges[:10])
            x = jnp.concatenate([nodes,received_edges,sent_edges],axis=1) #concatenates more feature to each node
            #print("Nodes:Received Edges",x[:10])
        
            for i, lyr in enumerate(self.fv_mlp):
            
                x = lyr(x)
                if (i != len(self.fv_mlp) - 1):
                    x = self.fv_activation(x)
           
            #x=jax.nn.hard_tanh(nodes+x)
            #print("Updated_Node")
            #print(x[:10])
            return x

        #fb
        def initial_edge_emb_fn(edges, senders, receivers, globals_):#fb
            x = edges
            #print("fb")
            #print("Edges",x[:10])
            for i, lyr in enumerate(self.fb_mlp):
                x = lyr(x)
                if (i != len(self.fb_mlp) - 1):
                    x = self.fb_activation(x)
            #print("Initial edge emb")
            #print(x[:10])
            
            return x
    
        #fa
        def initial_node_emb_fn(nodes, sent_edges, received_edges, globals_): #fa
            #print("fa")
            #print(nodes[:10])
            x=nodes  
            #print("nodes",x[:10])
            #x=self.fa_mlp_normalize_activation(x,axis=0)
            for i, lyr in enumerate(self.fa_mlp):
                x = lyr(x)
                if (i != len(self.fa_mlp) - 1):
                    x = self.fa_activation(x)
            #pri
            #print("Initial Node emb",x[:10])
            
            return x

        #MLP1
        def node_to_pi_fn(nodes):
            #Final output is pi( scaler)
            x = nodes
            #print("ff")
            #print(x)
            for i, lyr in enumerate(self.MLP1_mlp):
                x = lyr(x)
                x = self.MLP1_activation(x)
            #Normalize
            x=self.MLP1_normalize_activation(x,axis=0)
            return x

        myVijNet = Policy_Net(node_to_prob_fn=node_to_pi_fn,
                  initial_edge_embed_fn=initial_edge_emb_fn,
                  initial_node_embed_fn=initial_node_emb_fn,
                  update_edge_fn=update_edge_fn,
                  update_node_fn=update_node_fn,N=self.message_passing_steps)
        
       
        #MLP2x and MLP2y
        def Displace_node(G):
            """Returns [[mu_x ]
                        [mu_y ]]"""
            nodes, edges, receivers, senders, globals_, n_node, n_edge = G
            sum_n_node = tree.tree_leaves(nodes)[0].shape[0]

            #Aggregating edges 
            edges_sent_attributes = tree.tree_map(
                    lambda e: utils.segment_mean(e, senders, sum_n_node), edges)
            edges_received_attributes = tree.tree_map(
                    lambda e: utils.segment_mean(e, receivers, sum_n_node),edges)
    
            f_vec= jnp.concatenate([nodes,edges_sent_attributes,edges_received_attributes],axis=1) #concatenates more feature to each node
            
            #
            #Pass f_vec to MLP2
            
            x=f_vec
            for i, lyr in enumerate(self.MLP2_mlp):
                x = lyr(x)
                x = self.MLP2_activation(x)
    
                
            r_vec=0.01*self.sigma*x 
            r_vec=r_vec-jnp.mean(r_vec,axis=0)
            return r_vec 
            
        G,node_probs=myVijNet(graphs)                             
        Mux_Muy=Displace_node(G)     
        
        return G, node_probs, Mux_Muy
