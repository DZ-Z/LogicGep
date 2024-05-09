import numpy as np 
import pandas as pd
import multiprocessing
import operator
from deap import creator, base, tools
from operator import eq
from pre_RG import Regulators
import deap
from HallOfFame import *
from Gep_simple import gep_simple
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

def iterativeKmeans(data, d=3):  
    data = np.array(data)    
    while d > 0:  
        data = np.reshape(data, (-1,1))
        clusters = pow(2, d) 
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(data)   
        data = kmeans.cluster_centers_[kmeans.labels_] 
        d = d - 1  
    boolVal = kmeans.cluster_centers_[0,0] > kmeans.cluster_centers_[1,0] 
    centers = np.array([int(boolVal), int(not boolVal)])     
    return pd.Series(centers[kmeans.labels_].tolist())


def mlp2(raw_dataIn,y,list1):
    min_max_scaler = preprocessing.MinMaxScaler()
    # y = min_max_scaler.fit_transform(y) # if necessary
    y=y.ravel()
    X = np.array(raw_dataIn[list(set(list1))])
    # X = min_max_scaler.fit_transform(X) # if necessary
    fit1 = MLPRegressor(hidden_layer_sizes=(50, 15), activation='relu', solver='adam', alpha=0.01, max_iter=15)
    fit1.fit(X, y)
    pred1_train = fit1.predict(X)
    mse_1 = mean_squared_error(pred1_train, y)
    return mse_1


def mainn(target,Regulators_sets,data_Out,Input_data,binary_data,raw_dataIn,raw_dataOut,ss,pre_RG=True):

    if pre_RG:
        Regulators_sets=Regulators(target,binary_data)
        Input_data = Input_data[Regulators_sets].values.tolist()
    else:
        Input_data = Input_data[Regulators_sets].values.tolist()

    Out_data=data_Out[target].tolist()

    y = np.array(raw_dataOut[target].tolist()).reshape(len(raw_dataOut[target]), 1)

    pset = gep.PrimitiveSet('Main', input_names=Regulators_sets)
    pset.add_function(operator.and_, 2)
    pset.add_function(operator.or_, 2)
    pset.add_function(operator.not_, 1)

    creator.create("FitnessMin", base.Fitness, weights=(1,-1,1))  # to maximize the objective (fitness)
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

    h = 5  # head length
    n_genes = 1   # number of genes in a chromosome
    toolbox = gep.Toolbox()
    toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
    toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=None)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # compile utility: which translates an individual into an executable function (Lambda)
    toolbox.register('compile', gep.compile_, pset=pset)

    def evaluate(individual):
        """Evalute the fitness of an individual"""

        for gene in individual:
            input_variables=gene.kexpression
        list1= ['' + item.name + '' for item in input_variables]

        for item in list1[:]:
            if item == 'and_' or item == 'or_' or item == 'not_':
                list1.remove(item)
        n_regulators = len(set(list1))
        func= toolbox.compile(individual)
        n_correct=sum(func(*pIn) ==pOut for pIn, pOut in zip(Input_data, Out_data))
        chen=1
        return (n_correct, n_regulators,chen)

    def evaluate2(individual):
        """Evalute the fitness of an individual"""

        for gene in individual:
            input_variables = gene.kexpression
        list1 = ['' + item.name + '' for item in input_variables]
        for item in list1[:]:
            if item == 'and_' or item == 'or_' or item == 'not_':
                list1.remove(item)
        n_regulators = len(set(list1))
        func = toolbox.compile(individual)
        n_correct = sum(func(*pIn) == pOut for pIn, pOut in zip(Input_data, Out_data))
        mlp_loss = mlp2(raw_dataIn, y, list1)
        return (n_correct, n_regulators,mlp_loss)


    toolbox.register('evaluate', evaluate)
    toolbox.register('evaluate2', evaluate2)
    toolbox.register('select', tools.selTournament)
    ## general mutations whose aliases start with 'mut'
    # We can specify the probability for an operator with the .pbs property
    toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=2 / (2 * h + 1))
    toolbox.pbs['mut_uniform'] = 0.5
    # Alternatively, assign the probability along with registration using the pb keyword argument.
    toolbox.register('mut_invert', gep.invert, pb=0.5)
    toolbox.register('mut_is_ts', gep.is_transpose, pb=0.5)
    toolbox.register('mut_ris_ts', gep.ris_transpose, pb=0.5)
    toolbox.register('mut_gene_ts', gep.gene_transpose, pb=0.5)
    ## general crossover whose aliases start with 'cx'
    toolbox.register('cx_1p', gep.crossover_one_point, pb=0.5)
    toolbox.register('cx_2p', gep.crossover_two_point, pb=0.5)
    toolbox.register('cx_gene', gep.crossover_gene, pb=0.5)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    n_pop = 50
    n_gen = 200
    elites=10
    pop = toolbox.population(n=n_pop)
    hof = HallOfFame(5)   # only record the best individual ever found in all generations
    # start evolution
    pop2, log = gep_simple(pop, toolbox,
                              n_generations=n_gen, n_elites=elites,
                              stats=stats, hall_of_fame=hof, verbose=False)

    hof_sort=sorted(pop2, key=lambda ind: (-int(ind.fitness.values[0]), ind.fitness.values[2]))
    hof_sort2=[gep.simplify(ind) for ind in hof_sort]
    print(hof_sort2)

    symplified_best=hof_sort2[:1]
    print(symplified_best)
    with open(ss, 'a') as f:
        f.write(f'{target} = {symplified_best}\n')

if __name__ == "__main__":
    raw_data = pd.read_csv("./Data/rawdata.tsv", sep="\t", decimal=",")
    raw_data = raw_data.apply(pd.to_numeric) 
    raw_data= raw_data.dropna()
    print(raw_data)
    binary_data = raw_data.apply(iterativeKmeans, axis=0)
    print(binary_data)
    Regulators_sets = list(binary_data.columns)

    rows, cols = raw_data.shape

    dataIn=binary_data.loc[0:rows-2]
    dataOut=binary_data.loc[1:rows-1]
    data_In=dataIn==1
    data_Out=dataOut==1
    Input_data=data_In[list(dataIn.columns)].values.tolist()

    raw_dataIn = raw_data.loc[0:rows - 2]
    raw_dataOut = raw_data.loc[1:rows - 1]
    ss= "result.tsv"
    open(ss, 'w')
    list1=[(x1,Regulators_sets,data_Out,data_In,binary_data,raw_dataIn,raw_dataOut,ss,False)for x1 in list(binary_data.columns)]
    processes = [ multiprocessing.Process(target= mainn, args=[name, count, data_Out, Input_data, binary_data,raw_dataIn,raw_dataOut, ss, pre_RG])
    for name, count,data_Out,Input_data, binary_data,raw_dataIn,raw_dataOut,ss,pre_RG in list1]

    for p in processes:
        p.start()
    for p in processes:
        p.join()







