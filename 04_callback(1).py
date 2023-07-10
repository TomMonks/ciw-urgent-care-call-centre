#!/usr/bin/env python
# coding: utf-8

# # CiW Implementation of the 111 call centre
# 
# After a patient has spoken to a call operator their priority is triaged.  It is estimated that 40% of patients require a callback from a nurse.  There are 9 nurses available.  A nurse patient consultation has a Uniform distribution lasting between 10 and 20 minutes.
# 
# ![111 image](../../imgs/111_call_system.png "111 system")
# 
# **Task**
# 
# * Add a new decision variable to `Experiment` call `n_nurses`.
# * Create a second `simpy.Resource` called `nurses` and add it to the simulation model.
# * Modify the logic of `service` so that 40% of patients are called back.
# * Collect results and estimate the waiting time for a nurse callback and a nurse utilisation.
#  
# **Hints**
# 
# * Use the classes added called Uniform and Bernoulli distribution classes.
# * Not all patients will see a nurse - use the Bernoulli distribution to sample a True or False value.
# * The logic for taking a nurse resource and then undergoing a nurse consultation is the same the operator process.
# 

# ## 1. Imports

# In[1]:


import numpy as np
import pandas as pd
import ciw
import itertools

print(ciw.__version__)


# ## 2. Notebook level variables, constants, and default values
# 
# A useful first step when setting up a simulation model is to define the base case or as-is parameters.  Here we will create a set of constant/default values for our `Experiment` class, but you could also consider reading these in from a file.

# In[2]:


# default resources
N_OPERATORS = 13

# number of nurses available
N_NURSES = 9

# default mean inter-arrival time (exp)
MEAN_IAT = 100.0 / 60.0

## default service time parameters (triangular)
CALL_LOW = 5.0
CALL_MODE = 7.0
CALL_HIGH = 10.0

# nurse distribution parameters
NURSE_CALL_LOW = 10.0
NURSE_CALL_HIGH = 20.0

CHANCE_CALLBACK = 0.4

# Seeds for arrival and service time distributions (for repeatable single run)
ARRIVAL_SEED = 42
CALL_SEED = 101

# additional seeds for new activities
CALLBACK_SEED = 1966
NURSE_SEED = 2020

# Boolean switch to simulation results as the model runs
TRACE = False

# run variables
RESULTS_COLLECTION_PERIOD = 1000


# ## 3. Experiment class
# 
# We will modify the experiment class to include new results collection for the additional nurse process. 
# 
# 1. Modify the __init__ method to accept additional parameters: `chance_callback`, `nurse_call_low`, `nurse_call_high`, `callback_seed`, `nurse_seed`. Remember to include the default values for these parameters.
# 2. Store parameters in the class and create new distributions.
# 3. Add variables to support KPI calculation to the `results` dictionary for `nurse_waiting_times` and `total_nurse_call_duration`

# In[3]:


class Experiment:
    '''
    Parameter class for 111 simulation model
    '''
    def __init__(self, n_operators=N_OPERATORS, n_nurses=N_NURSES, 
                 mean_iat=MEAN_IAT, call_low=CALL_LOW, call_mode=CALL_MODE, 
                 call_high=CALL_HIGH, chance_callback=CHANCE_CALLBACK, 
                 nurse_call_low=NURSE_CALL_LOW, nurse_call_high=NURSE_CALL_HIGH,
                 random_seed=None):
        '''
        The init method sets up our defaults. 
        '''
        self.n_operators = n_operators
        
        # store the number of nurses in the experiment
        self.n_nurses = n_nurses
        
        # arrival distribution
        self.arrival_dist = ciw.dists.Exponential(mean_iat)
        
        # call duration 
        self.call_dist = ciw.dists.Triangular(call_low, call_mode, call_high)
        
        # duration of call with nurse     
        self.nurse_dist = ciw.dists.Uniform(nurse_call_low, nurse_call_high)
        
        # prob of call back
        self.chance_callback = chance_callback
                
        # initialise results to zero
        self.init_results_variables()
        
    def init_results_variables(self):
        '''
        Initialise all of the experiment variables used in results 
        collection.  This method is called at the start of each run
        of the model
        '''
        # variable used to store results of experiment
        self.results = {}
        self.results['waiting_times'] = []
        
        # total operator usage time for utilisation calculation.
        self.results['total_call_duration'] = 0.0
        
        # nurse sub process results collection
        self.results['nurse_waiting_times'] = []
        self.results['total_nurse_call_duration'] = 0.0


# ## 4. Model code

# In[4]:


def get_model(args):
    '''
    Build a CiW model using the arguments provided.
    
    Params:
    -----
    args: Experiment
        container class for Experiment. Contains the model inputs/params
        
    Returns:
    --------
    ciw.network.Network
    '''
    model = ciw.create_network(arrival_distributions=[args.arrival_dist,
                                                      ciw.dists.NoArrivals()],
                               service_distributions=[args.call_dist,
                                                      args.nurse_dist],
                               routing=[[0.0, 0.4],
                                        [0.0, 0.0]],
                               number_of_servers=[args.n_operators,
                                                  args.n_nurses])
    return model


# ## 5. Model wrapper functions
# 
# Modifications to make to the `single_run` function:
# 
# * Create and the nurses resource to the experiment
# * After the simulation is complete calculate the mean waiting time and mean nurse utilisation.
# 
# 
# **Hints:**
# 
# * To create a nurse resource and assign it to the experiment you can use the following code:
# 
# ```python
# experiment.nurses = simpy.Resource(env, capacity=experiment.n_nurses)
# ```
# 
# * You do not need to make any modifications to the `multiple_replications` function

# In[5]:


def single_run(experiment, rc_period=RESULTS_COLLECTION_PERIOD, 
               random_seed=None):
    '''
    Conduct a single run of the simulation model.
    
    Params:
    ------
    args: Scenario
        Parameter container
        
    random_seed: int
        Random seed to control simulation run.
    '''
    
    # results dictionary.  Each KPI is a new entry.
    run_results = {}
    
    # random seed
    ciw.seed(random_seed)

    # parameterise model
    model = get_model(experiment)

    # simulation engine
    sim_engine = ciw.Simulation(model)
    
    # run the model
    sim_engine.simulate_until_max_time(rc_period)
    
    # return processed results for run.
    
    # get all results
    recs = sim_engine.get_all_records()
    
    # operator service times
    op_servicetimes = [r.service_time for r in recs if r.node==1]
    # nurse service times
    nurse_servicetimes = [r.service_time for r in recs if r.node==2]
    
    # operator and nurse waiting times
    op_waits = [r.waiting_time for r in recs if r.node==1]
    nurse_waits = [r.waiting_time for r in recs if r.node==2]
    
    # mean measures
    run_results['01_mean_waiting_time'] = np.mean(op_waits)
        
    # end of run results: calculate mean operator utilisation
    run_results['02_operator_util'] = \
        (sum(op_servicetimes) / (rc_period * experiment.n_operators)) * 100.0
    
    # end of run results: nurse waiting time
    run_results['03_mean_nurse_waiting_time'] = np.mean(nurse_waits)
    
    # end of run results: calculate mean nurse utilisation
    run_results['04_nurse_util'] = \
        (sum(nurse_servicetimes) / (rc_period * experiment.n_nurses)) * 100.0
    
    # return the results from the run of the model
    return run_results


# In[7]:


def multiple_replications(experiment, 
                          rc_period=RESULTS_COLLECTION_PERIOD,
                          n_reps=5):
    '''
    Perform multiple replications of the model.
    
    Params:
    ------
    experiment: Experiment
        The experiment/paramaters to use with model
    
    rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.  
        the number of minutes to run the model to collect results

    n_reps: int, optional (default=5)
        Number of independent replications to run.
        
    Returns:
    --------
    pandas.DataFrame
    '''

    # loop over single run to generate results dicts in a python list.
    results = [single_run(experiment, rc_period) for rep in range(n_reps)]
        
    # format and return results in a dataframe
    df_results = pd.DataFrame(results)
    df_results.index = np.arange(1, len(df_results)+1)
    df_results.index.name = 'rep'
    return df_results


# In[8]:


TRACE = False
default_scenario = Experiment()
results = multiple_replications(default_scenario)
results


# ## 6. Multiple experiments
# 
# > No modifications are needed to code in this section.
# 
# The `single_run` and `multiple_replications` wrapper functions for the model and the `Experiment` class mean that is very simple to run multiple experiments using replication analysis.  We will define three new functions for running multiple experiments:
# 
# * `get_experiments()` - this will return a python dictionary containing a unique name for an experiment paired with an `Experiment` object
# * `run_all_experiments()` - this will loop through the dictionary, run all experiments and return combined results.
# * `experiment_summary_frame()` - take the results from each scenario and format into a simple table.

# In[9]:


def get_experiments():
    '''
    Creates a dictionary object containing
    objects of type `Experiment` to run.
    
    Returns:
    --------
    dict
        Contains the experiments for the model
    '''
    experiments = {}
    
    # base case
    # we will sync scenarios by using seeds
    experiments['base'] = Experiment(random_seed=42)
    
    # +1 extra capacity
    experiments['operators+1'] = Experiment(random_seed=42,
                                            n_operators=N_OPERATORS+1)
    
    return experiments


# In[14]:


def run_all_experiments(experiments, rc_period=RESULTS_COLLECTION_PERIOD):
    '''
    Run each of the scenarios for a specified results
    collection period and replications.
    
    Params:
    ------
    experiments: dict
        dictionary of Experiment objects
        
    rc_period: float
        model run length
    
    '''
    print('Model experiments:')
    print(f'No. experiments to execute = {len(experiments)}\n')

    experiment_results = {}
    for exp_name, experiment in experiments.items():
        
        print(f'Running {exp_name}', end=' => ')
        results = multiple_replications(experiment, rc_period)
        print('done.\n')
        
        #save the results
        experiment_results[exp_name] = results
    
    print('All experiments are complete.')
    
    # format thje results
    return experiment_results
                    


# In[15]:


# get the experiments
experiments = get_experiments()

#run the scenario analysis
experiment_results = run_all_experiments(experiments)


# In[16]:


def experiment_summary_frame(experiment_results):
    '''
    Mean results for each performance measure by experiment
    
    Parameters:
    ----------
    experiment_results: dict
        dictionary of replications.  
        Key identifies the performance measure
        
    Returns:
    -------
    pd.DataFrame
    '''
    columns = []
    summary = pd.DataFrame()
    for sc_name, replications in experiment_results.items():
        summary = pd.concat([summary, replications.mean()], axis=1)
        columns.append(sc_name)

    summary.columns = columns
    return summary


# In[17]:


# as well as rounding you may want to rename the cols/rows to 
# more readable alternatives.
summary_frame = experiment_summary_frame(experiment_results)
summary_frame.round(2)

