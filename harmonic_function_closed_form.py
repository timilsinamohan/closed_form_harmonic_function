__author__ = 'mohan'
from scipy.sparse import lil_matrix
from rescal import rescal_als

import networkx as nx
from networkx.algorithms import bipartite
from scipy import sparse
import pandas as pd
from sklearn.metrics import roc_auc_score

from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KNeighborsClassifier
from scipy import linalg
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
import time

np.set_printoptions(precision=3,suppress=True)
def create_tensors():

    ##binding drugs function and genes by creating dictionary ###
    ent = set()
    cnt = 0
    for tumor_nodes in tumor_nodes_col:
        ent.add(tumor_nodes)
        ent.add(gene_nodes_col[cnt])
        cnt+=1

    cnt = 0
    for drug_nodes in drug_nodes_col:
        ent.add(drug_nodes)
        ent.add(drug_gene_nodes_col[cnt])
        cnt+=1

    ent = list(ent)
    ##create dictionary of nodes##
    nodes_dict ={}
    cnt = 0
    for n in ent:
        nodes_dict[n] = cnt
        cnt+=1

   
    e = len(ent)
    m = 2 ##For 2 different relationship##
    X = [lil_matrix((e,e)) for i in range(m)]

    #print tumor_nodes_col
    tumor_gene_edgelist = zip(tumor_nodes_col,gene_nodes_col)
    
    #### tumor gene bipartite graph nodes and edges##
    B = nx.Graph()
    B.add_nodes_from(tumor_nodes_col,bipartite = 0)
    B.add_nodes_from(gene_nodes_col,bipartite = 1)
    B.add_edges_from(tumor_gene_edgelist)
    drug_gene_edgelist = zip(drug_gene_nodes_col,drug_nodes_col)
    
    #### drug gene bipartite graph nodes and edges##
    B = nx.Graph()
    B.add_nodes_from(drug_gene_nodes_col,bipartite = 0)
    B.add_nodes_from(drug_nodes_col,bipartite = 1)
    B.add_edges_from(drug_gene_edgelist)
    genes_only_nodes= set()
    genes_name_only = set()
    tumor_only = set()

    ###creating hasGene networks####
    cnt_has_gene = 0
    for u,v in tumor_gene_edgelist:
        X[0][nodes_dict.get(u),nodes_dict.get(v)] = 1
        genes_only_nodes.add(nodes_dict.get(v))
        genes_name_only.add(v)
        tumor_only.add(nodes_dict.get(u))
        cnt_has_gene+=1
    
    print "Number of hasGene Relationship:",cnt_has_gene
    
     ###creating drugGene networks####
    cnt_drug_gene_action = 0
    for u,v in drug_gene_edgelist:
        X[1][nodes_dict.get(v),nodes_dict.get(u)] = 1
        genes_only_nodes.add(nodes_dict.get(u))
        cnt_drug_gene_action += 1
      

    #print "created tumor gene tensor"
    gene_index = list(genes_only_nodes)
    gene_names = list(genes_name_only)
 
    return X,gene_names,gene_index,nodes_dict



def ground_truth_node_labels(gene_names):
    drug_function_edgelist = zip(drug_target_functions,drug_gene_nodes_col)
    labels= {}

    for gene_name,func in drug_function_edgelist:
        labels.setdefault(gene_name, []).append(func)

    G = nx.DiGraph(labels)
    B = nx.Graph()
    lab=[]
    genes=[]
    edges =[]
    for u,v in G.edges:
        lab.append(u)
        genes.append(v)
        edges.append([u,v])

    B.add_nodes_from(lab,bipartite = 0)
    B.add_nodes_from(genes,bipartite = 1)
    B.add_edges_from(edges)
   
    Ground_Truth_Matrix = nx.algorithms.bipartite.biadjacency_matrix(B,row_order= labels.keys(),
                                                                     column_order=gene_names)

    return Ground_Truth_Matrix,labels.keys()



def gene_interaction_network(network,gene_nodes):
    G = nx.read_edgelist(network, delimiter= " ",
                         nodetype=str,
                         data=(('weight',float),))
    
    Gene_Gene_Adj_mat = nx.adjacency_matrix(G, nodelist=gene_nodes)
    GG = nx.from_scipy_sparse_matrix(Gene_Gene_Adj_mat)
    return GG




def get_embeddings(Tenc):
    feature_vec,R, fit, itr, exectimes = rescal_als(Tenc,250,
                                                    init='nvecs',
                                                    conv=1e-2,
                                                    lambda_A=0.1,
                                                    lambda_R= 0.1 
                                                   )
    
    return feature_vec



def get_graph(labels,EMB,EXP,train_nodes,test_nodes):
    order = np.append(train_nodes,test_nodes)
    k_range = range(1,12)
    param_grid = dict(n_neighbors = k_range)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn,param_grid, cv = 10, scoring = "accuracy")
    grid.fit(gene_feature,labels)
    GF = kneighbors_graph(gene_feature,grid.best_params_['n_neighbors'], mode='connectivity',include_self=False)
    G1 = nx.from_numpy_matrix(GF.A)
    L1 = nx.laplacian_matrix(G1,nodelist = order, weight='weight')
    degrees_g1 = GF.sum(axis=0).A[0]
    degrees_g1[degrees_g1==0] += 1  # Avoid division by 0
    D_g1 = sparse.diags((1.0/degrees_g1),offsets=0)
    L1_norm = D_g1.dot(L1).tolil()

    G2 = gene_interaction_network(network,gene_names)
    G2 = nx.adjacency_matrix(G2)
    degrees_g2 = G2.sum(axis=0).A[0]
    degrees_g2[degrees_g2==0] += 1  # Avoid division by 0
    D_g2 = sparse.diags((1.0/degrees_g1),offsets=0)
    L2 = nx.laplacian_matrix(get_network, nodelist=order, weight='weight')
    L2_norm = D_g2.dot(L2).tolil()
    graph_matx = np.add(L1_norm*EMB,L2_norm*EXP)

    return graph_matx




def get_harmonic_score(train_nodes,mask_labels,graph):
    r,c = graph.shape
    l = len(train_nodes)
    u = r - len(train_nodes)
    Lll = graph[0:l,0:l]
    Llu = graph[0:l,l:r]
    Lul = graph[l:r,0:l]
    Luu = graph[l:r,l:r]
    yl =  mask_labels[train_nodes]
    fu = -linalg.pinv(Luu.A).dot(Lul.A).dot(yl)
    return fu

def get_innerfold(train_nodes,test_nodes):
    mask_labels = GT.copy()
    mask_labels[test_nodes] = 0
    graph = get_graph(mask_labels,EMB,EXP,train_nodes,test_nodes)
    ####call harmonic function###
    scrs = get_harmonic_score(train_nodes,mask_labels,graph)

    return scrs

    
if __name__ == '__main__':
    # load data
    df1 = pd.read_csv("data/tumor_gene_hasGene_data.csv",sep = "\t",names = ["tumor", "gene"])
    df2 = pd.read_csv("data/drugs_gene_copy.txt",sep = "\t",names = ["gene", "drugs","function"])

    tumor_nodes_col = df1["tumor"]
    gene_nodes_col =df1["gene"]

    drug_nodes_col = df2["drugs"]
    drug_gene_nodes_col =df2["gene"]
    drug_target_functions = df2["function"]
    
    ##creating tensors for factorization for drug-gene-tumor multi-graph###

    Tenc, gene_names, gene_idx, node_dictionary =  create_tensors() 
    
    ###getting gene embeddings#######
    
    features_vector_for_genes = get_embeddings(Tenc) 
    gene_feature = features_vector_for_genes[gene_idx,:]
    #####################################################
    
    gene_names=[]
    for id in gene_idx:
        gene_names.append(node_dictionary.keys()[node_dictionary.values().index(id)])

    ground_truth,labels_keys = ground_truth_node_labels(gene_names)
    
    ### types of labels available "blocker","antagonist","agonist","activator","inhibitor","channel blocker",
    ## "binder"
   
    ##"Positive Labels:"###
    GT = ground_truth[labels_keys.index("blocker"),:].A[0]
   
    ##Negative Labels###
    idx = np.where(GT==0)
    GT[idx] = -1


    network = "data/combined_scores.txt"
        
    ##percentage of the labelled nodes###

    ##########################
    get_network = gene_interaction_network(network,gene_names)
    
    ########### To Enable only embedding Network make the flag EMB = 1 and EXP = 0 and vice versa##
    ### EMB = Embedding Network
    ### EXP = Genetic Experimental Network
    EMB = 1
    EXP = 1
    ########################################
    labeled_percentage = 0.3
    FOLDS = 10
    roc_score = np.zeros(FOLDS)
    IDX = list(range(len(GT)))
    cnt = 0
    rs = ShuffleSplit(n_splits=FOLDS, test_size=(1-labeled_percentage), random_state=0)

    for train_split, test_split in rs.split(IDX):
        start = time.time()
        results = get_innerfold(train_split,test_split)
        roc_score[cnt] = roc_auc_score(GT[test_split], results)
        cnt +=1
        print "Time taken to complete the fold:", time.time()-start
    print "AUC ROC in each folds:",roc_score
    print('AUC-ROC Test Mean / Std: %f / %f' % (roc_score.mean(), roc_score.std()))

