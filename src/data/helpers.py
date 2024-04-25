import json
import networkx as nx
import pandas as pd
import torch
import os

### old function - do not use
def CleanGraphFromUnconnectedElements(elements,edges):
    ### filter out unconnected elements
    edge_set = set()
    for e in edges:
        edge_set.add(e[0])
        edge_set.add(e[1])
    new_elements = [e for e in elements if e['unique_id'] in edge_set]
    return new_elements

def OneHotDynamic(frame, col_name, amount, bin_storage):
    col = pd.to_numeric(frame[col_name], errors='coerce')
    binned,bins = pd.qcut(col, amount, precision=3,retbins=True)
    bin_storage.update({col_name:bins.tolist()})
    dummies = pd.get_dummies(binned, prefix=col_name)
    return dummies

def OneHotFixed(frame, col_name, amount, bin_storage):
    col = pd.to_numeric(frame[col_name], errors='coerce')
    span = col.max() - col.min()
    bins = [(col.min()+ i*(span/amount)) for i in range(amount)]
    bins.append(col.max()+0.1)
    binned,bins = pd.cut(col, bins, precision=3,retbins=True, include_lowest=True)
    bin_storage.update({col_name:bins.tolist()})
    dummies = pd.get_dummies(binned, prefix=col_name)
    return dummies

def OneHotSpecific(frame, col_name, bins, bin_storage):
    col = pd.to_numeric(frame[col_name], errors='coerce')
    bins.insert(0,col.min())
    bins.append(col.max()+0.1)
    binned,bins = pd.cut(col, bins, precision=3,retbins=True, include_lowest=True)
    bin_storage.update({col_name:bins.tolist()})
    dummies = pd.get_dummies(binned, prefix=col_name)


def OneHotFromStore(frame, col_name,  bin_storage):
        col = pd.to_numeric(frame[col_name], errors='coerce')
        bins = bin_storage[col_name]
        binned = pd.cut(col, bins, precision=3,retbins=False, include_lowest=True)
        dummies = pd.get_dummies(binned, prefix=col_name)
        #print ('checking dummies for column ' + col_name)
        CheckDummies(dummies)
        return dummies

def CheckDummies(dummies):
    for i in range(len(dummies)):
        if  dummies.loc[i].sum() != 1:
            print("item: "+ str(i)+ "doesnt fall into the bins" )

def CreateGraph(_elements, _edges):
    G = nx.Graph()
    for node in _elements:
        G.add_node(node['unique_id'])
    for edge in _edges:
        G.add_edge(edge[0],edge[1])
    return G

def GetConnectedGraph(graph):
    ### filter out unconnected elements
    largest_cc = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_cc)

def GetCOOMatrix(graph):
    element_ids = list(graph.nodes)
    from_indexes = []
    to_indexes = []
    for e in graph.edges:
        from_indexes.append(element_ids.index(e[0]))
        to_indexes.append(element_ids.index(e[1]))
    edge_index = torch.tensor([from_indexes,to_indexes],dtype = torch.int64)
    print("generated COO matrix:" + str(edge_index.shape))
    return edge_index

def GenerateOneHotEncoding(elements, bin_path):
    df = pd.DataFrame(elements)
    all_bins = {}
    '''
    with open(bin_path) as fp:
        all_bins = json.load(fp)
    #print (all_bins)
    
    BB_X_dim = OneHotFromStore(df,'BB_X_dim',all_bins)
    BB_Y_dim = OneHotFromStore(df,'BB_Y_dim',all_bins)
    BB_Z_dim = OneHotFromStore(df,'BB_Z_dim',all_bins)
    BB_volume = OneHotFromStore(df,'BB_volume',all_bins)
    solid_volume = OneHotFromStore(df,'solid_volume',all_bins)
    relative_height = OneHotFromStore(df,'relative_height',all_bins)
    num_of_components = OneHotFromStore(df,'num_of_components',all_bins)
    '''

    BB_X_dim = OneHotDynamic(df,'BB_X_dim',all_bins)
    BB_Y_dim = OneHotDynamic(df,'BB_Y_dim',all_bins)
    BB_Z_dim = OneHotDynamic(df,'BB_Z_dim',all_bins)
    BB_volume = OneHotDynamic(df,'BB_volume',all_bins)
    solid_volume = OneHotDynamic(df,'solid_volume',all_bins)
    relative_height = OneHotDynamic(df,'relative_height',all_bins)
    num_of_components = OneHotDynamic(df,'num_of_components',all_bins)
    encoding = pd.concat([BB_X_dim, BB_Y_dim,BB_Z_dim,BB_volume,solid_volume, relative_height,num_of_components],axis=1)
    ### unused features , df['largest_dim'],df['smallest_dim'], num_of_components,df['plane_Z_direction'], relative_height
    ## print ('encoding shape ' + str(encoding.shape))
    return encoding




def Testing(out):
    return out