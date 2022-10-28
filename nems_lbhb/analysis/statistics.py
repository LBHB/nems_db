import numpy as np


# hierarchachal bootstrap, see: cite biorxiv paper
def get_bootstrapped_sample(variable, variable2=None, metric=None, even_sample=False, nboot=1000):
    '''
    Adapted from Saravanan et al. 2020 arXiv

    This function performs a hierarchical bootstrap on the data present in 'variable'.
    This function assumes that the data in 'variable' is in the format of a dict where
    the keys represent the higher level (e.g. animal/recording site) and
    the values (1D arrays) represent repetitions/observations within that level (e.g. neurons).

    Only set up to handle two-level data right now 08.21.2020, CRH

    CRH 01.30.2021 - modified to allow computing corr. coefs hierarchically
        set variable1 to the dict of x values, variable2 to the dict of y values, and metric to corrcoef
    '''
    bootstats = np.zeros(nboot)
    for i in np.arange(nboot):
        temp = []
        temp2 = []
        num_lev1 = len(variable.keys())        # n animals
        num_lev2 = max([variable[n].shape[0] for n in variable.keys()]) # min number of observations sampled for an animal
        rand_lev1 = np.random.choice(num_lev1, num_lev1)
        lev1_keys = np.array(list(variable.keys()))[rand_lev1]
        for k in lev1_keys:
            # for this animal, how many obs to choose from?
            this_n_range = variable[k].shape[0]
            if even_sample:
                rand_lev2 = np.random.choice(this_n_range, num_lev2, replace=True)
            else:
                rand_lev2 = np.random.choice(this_n_range, this_n_range, replace=True)
            
            temp.extend(variable[k][rand_lev2].tolist())   # k is saying which animal, rand_lev2 are the observations from this animal
            if variable2 is not None:
                temp2.extend(variable2[k][rand_lev2].tolist())

        #Note that this is the step at which actual computation is performed. In all cases for these simulations
        #we are only interested in the mean. But as elaborated in the text, this method can be extended to 
        #several other metrics of interest. They would be computed here:
        #bootstats[i] = np.mean(np.concatenate(temp))
        if metric=='corrcoef':
            bootstats[i] = np.corrcoef(temp, temp2)[0, 1]
        elif metric=='median':
            bootstats[i] = np.median(temp)
        elif metric=='mean':
            bootstats[i] = np.mean(temp)
        elif metric is None:
            bootstats[i] = np.mean(temp)
        else:
            raise ValueError(f"No code implemented for method: {metric} yet!")

    return bootstats


def get_direct_prob(sample1, sample2, twosided=False):
    '''
    get_direct_prob Returns the direct probability of items from sample2 being
    greater than or equal to those from sample1.
       Sample1 and Sample2 are two bootstrapped samples and this function
       directly computes the probability of items from sample 2 being greater
       than or equal to those from sample1. Since the bootstrapped samples are
       themselves posterior distributions, this is a way of computing a
       Bayesian probability. The joint matrix can also be returned to compute
       directly upon.
    '''
    joint_low_val = min([min(sample1),min(sample2)])
    joint_high_val = max([max(sample1),max(sample2)])
    
    p_joint_matrix = np.zeros((100,100))
    p_axis = np.linspace(joint_low_val,joint_high_val,num=100)
    edge_shift = (p_axis[2] - p_axis[1])/2
    p_axis_edges = p_axis - edge_shift
    p_axis_edges = np.append(p_axis_edges, (joint_high_val + edge_shift))

    #Calculate probabilities using histcounts for edges.

    p_sample1 = np.histogram(sample1,bins=p_axis_edges)[0]/np.size(sample1)
    p_sample2 = np.histogram(sample2,bins=p_axis_edges)[0]/np.size(sample2)

    #Now, calculate the joint probability matrix:

    for i in np.arange(np.shape(p_joint_matrix)[0]):
        for j in np.arange(np.shape(p_joint_matrix)[1]):
            p_joint_matrix[i,j] = p_sample1[i]*p_sample2[j]
            
    #Normalize the joint probability matrix:
    p_joint_matrix = p_joint_matrix/np.sum(p_joint_matrix)
    
    #Get the volume of the joint probability matrix in the upper triangle:
    p_test = np.sum(np.triu(p_joint_matrix))

    #Convert one-sided p-value to two-sided
    if twosided:
        new_p = 2 * (1 - p_test)
        if new_p > 1:
            # one-sided value was in the "wrong" direction
            new_p = p_test * 2
        p_test = new_p
    
    return p_test, p_joint_matrix


def df_to_site_dict(df, column_names=None):
    """Convert dataframe with cellid index to dicts with one key per site, one dict per column.

    Examples
    --------
    >>> df = nems0.db.batch_comp(322, modelnames=['model_one', 'model_two', ...], stat='r_test')
    >>> dict_one, dict_two = df_to_site_dict(df, column_names=['model_one', 'model_two'])
    >>> boot_one = get_bootstrapped_sample(dict_one)
    >>> boot_two = get_bootstrapped_sample(dict_two)
    >>> p = get_direct_probability(boot_one, boot_two)

    """
    if column_names is None:
        column_names = df.columns.tolist()

    cellids = df.index.values.tolist()
    sites = list(set([c.split('-')[0] for c in cellids]))
    dicts = []
    for col in column_names:
        col_dict = {}
        for site in sites:
            site_idx = df[col].index.str.contains(site)
            site_vals = df[col].loc[site_idx].values
            col_dict[site] = site_vals
        dicts.append(col_dict)

    return dicts


def array_to_site_dict(array, cellids):
    """Split 1D array by referencing cellids, formatted as a dict with one key per site.

    Warnings
    --------
    Values in array must be in the same order as cellids
    (i.e. the value in array[0] corresponds to cellids[0]).

    Examples
    --------
    >>> df = nems0.db.batch_comp(322, modelnames=['model_one', 'model_two', ...], stat='r_test')
    >>> array = df['model_one'].values
    >>> cellids = df.index.values.tolist()
    >>> # array = ...  (compute stuff with array w/o changing ordering)
    >>> site_dict = array_to_site_dict(array, cellids)
    >>> boot = get_bootstrapped_sample(site_dict)

    """
    sites = list(set([c.split('-')[0] for c in cellids]))
    site_dict = {}
    for site in sites:
        site_match = [c.startswith(site) for c in cellids]
        site_idx = [i for i, m in enumerate(site_match) if m]
        site_dict[site] = array[site_idx]

    return site_dict


def arrays_to_p(array1, array2, cellids, boot_kwargs=None, twosided=False):
    """Compute direct probability for array2 >= array1."""
    boot_kwargs = boot_kwargs if boot_kwargs is not None else {'nboot': 1000}
    site_dict1 = array_to_site_dict(array1, cellids)
    site_dict2 = array_to_site_dict(array2, cellids)
    boot1 = get_bootstrapped_sample(site_dict1, variable2=None, **boot_kwargs)
    boot2 = get_bootstrapped_sample(site_dict2, variable2=None, **boot_kwargs)
    p, _ = get_direct_prob(boot1, boot2, twosided=twosided)

    results = (p, f'hierarch. boot., options: {boot_kwargs}')
    return results


def df_to_p(df, column_names=None, boot_kwargs=None, twosided=False):
    """Compute direct probability for df[col_name_2] >= df[col_name_1].

    Only the first two column names will be used (or the first two columns,
    if no column names are specified).

    """
    boot_kwargs = boot_kwargs if boot_kwargs is not None else {'nboot': 1000}
    site_dicts = df_to_site_dict(df, column_names)[:2]
    boots = [get_bootstrapped_sample(s, variable2=None, **boot_kwargs)
             for s in site_dicts]
    p, _ = get_direct_prob(*boots, twosided=twosided)

    results = (p, f'hierarch. boot., options: {boot_kwargs}')
    return results
