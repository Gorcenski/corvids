__author__ = 'Sean Wilner'
from findSolutionsWithManipBasis import *
from decimal import Decimal
from functools import reduce
from sympy import Matrix
import math, random, itertools, functools, collections

from mpl_toolkits.mplot3d import Axes3D
from sys import platform as sys_pf
if sys_pf == 'darwin':
     import matplotlib
     matplotlib.use("TkAgg")
else:
    import matplotlib
import matplotlib.pyplot as plt
import numpy as np


dio = Diophantine()

def multiprocessGetManipBases(basis_and_base_vec):
    basis, base_vec = basis_and_base_vec
    manip_basis, base_vec = getManipBasis(basis, base_vec)

    manip_base_vec = forced_neg_removal(manip_basis, base_vec)
    return (manip_basis, manip_base_vec)

def multiprocess_get_solution_space(min_score, max_score, num_samples, mean_and_variance,
                                 check_val=None, poss_vals=None, debug=True):
    '''
    A function call to mimic the one on the RecreateData object while remaining multiprocess compatible
        (can't pickle an object method within the object namespace)

    This function handles getting a valid initial solution (possibly with negative values) and a corresponding vector
        space of valid transformations to that solution.

    Typical users should never need to call this function as it is called from within the recreateData() method on
        the RecreateData object.
    :param min_score: minimum value in the range of possible values for the dataset
    :param max_score: maximum value in the range of possible values for the dataset
    :param num_samples: total number of samples to find solutions for
    :param mean_and_variance: a tuple containing the mean and variance of the dataset e.g. (mean,variance)
    :param check_val: values we want to assume the solution space MUST contain presented as either a single integer
                            or as a list of integers, or as a dictionary whose keys is the integers assumed to be in
                            the dataset with corresponding values of the number of times the integer is assumed to
                            appear in the dataset.  That is, this value lets the user check for specific values in
                            solutions.
    :param poss_vals: an iterable (eg list) containing all potential values to consider when constructing viable
                            datasets.  That is, this value lets the user remove given values from consideration for
                            solution spaces (by presenting all other potential values).
    :param debug: Boolean to indicate if print statements are allowed.
    :return: Either:
                            A 5-tuple containing an initial (potentially negative) solution, the basis for a vector space of
                            transformations, the two matrices used to calculate it, and the mean_and_variance tuple
                            passed as an argument if such a solution exists

             Or:
                            None if no solution exists
    '''
    if not poss_vals:
        poss_vals = xrange(min_score, max_score+1)

    mean, variance = mean_and_variance
    mean = int(round(mean * num_samples))
    # use this type casting to handle larger numbers with better precision
    variance = int(Decimal(str(variance * (num_samples - 1) * num_samples ** 2)))
    A = Matrix([[1, i, ((num_samples * i) - mean) ** 2] for i in poss_vals]).T

    if check_val:
        if isinstance(check_val,int):
            check_vals = [(check_val, 1)]
        elif isinstance(check_val, list):
            check_vals = zip(check_val, [1] * len(check_val))
        elif isinstance(check_val, dict):
            check_vals = [(val, num) for val, num in check_val.iteritems()]
        else:
            raise TypeError
        # further refactor: move the lambda to a utility function 
        [num_samples, mean, variance] = reduce(lambda x, y: [x[0] - y[1],
                                                             x[1] - y[0] * y[1],
                                                             x[2] - y[1] * (x[0] * y[0] - x[1]) ** 2],
                                               check_vals,
                                               [num_samples, mean, variance])

    b = Matrix([num_samples, mean, variance])
    basis = Matrix(dio.getBasis(A, b))
    try:
        base_vec = basis[-1,:]
        basis = basis[:-1,:]
        if debug:
            print("found potential at: " + str(mean_and_variance))
        return base_vec, basis, A, b, mean_and_variance
    except IndexError:
        print("Unable to find a potential solution")
        return None

class RecreateData:
    '''
    An object which conatins all the relevant information about a given set of summary statistics and allows methods
        to discover all potential solutions
    '''

    def __init__(self, min_score, max_score, num_samples, mean, variance, debug=True, mean_precision=0.0, variance_precision=0.0):
        self.simpleData = defaultdict(list)
        self.debug = debug

        self.min_score = min_score
        self.max_score = max_score
        self.absolute_min = min_score
        self.absolute_max = max_score
        self.poss_vals = range(self.absolute_min, self.absolute_max + 1)
        self.extended_poss_vals = None
        
        self.num_samples = num_samples
        self.mean = mean
        self.variance = variance
        self.un_mut_num_samples = num_samples
        self.un_mut_mean = mean
        self.un_mut_variance = variance
        
        self.sols = None
        self.mean_precision = mean_precision
        self.variance_precision = variance_precision
        
        # precompute at initialization for better code clarity and performance
        self.mean_range = xrange(int(math.ceil((self.mean - self.mean_precision) * self.num_samples)),
                                 int(math.floor((self.mean + self.mean_precision) * self.num_samples)) + 1)
        self.n_minus_1_n_squared = (self.num_samples - 1) * self.num_samples ** 2


    def set_ranges_with_possible_values(self, poss_vals):
        self.min_score = min(poss_vals)
        self.max_score = max(poss_vals)
        self.poss_vals = poss_vals


    def adjust_sol(self, solution, num_adjustments):
        if num_adjustments == 0:
            return solution
        new_sol = deepcopy(solution)
        new_sol.sort()
        num_adjusted = 0
        while num_adjusted < num_adjustments:
            correction = num_adjustments - num_adjusted
            items = list(set(new_sol))
            items.sort()
            if correction == 1:
                candidates = [item for item, count in collections.Counter(new_sol).items() if count > 1 and (item + 1) in self.poss_vals and (item - 1) in self.poss_vals]
                if len(candidates)==0:
                    return None
                potential_diff = candidates[0]
                new_sol.remove(potential_diff)
                new_sol.remove(potential_diff)
                new_sol.append(potential_diff + 1)
                new_sol.append(potential_diff - 1)
                new_sol.sort()
                return new_sol
            potential_diffs = []
            temp_items = []
            if len(new_sol)>len(items):
                for val in items:
                    if new_sol.count(val) == 1:
                        continue
                    temp_items.append(val)
            items.extend(temp_items)
            items.sort()
            for val_1, val_2 in itertools.combinations(items, 2):
                #since we sorted items, val_1 is the smaller one
                for size in xrange(1, min((max(self.poss_vals) - val_2), val_1 - min(self.poss_vals))):
                    if val_1 - size not in self.poss_vals or val_2 + size not in self.poss_vals:
                        continue
                    adjustment = (math.fabs(val_1 - val_2)*size + size**2)
                    if adjustment <= correction:
                        potential_diffs.append((adjustment, val_1, val_2, size))
            if len(potential_diffs) == 0:
                return
            potential_diffs.sort()
            potential_diff = potential_diffs[-1] #Greed algorithm here to choose the biggest adjustment possible
            num_adjusted += potential_diff[0]
            new_sol.remove(potential_diff[1])
            new_sol.remove(potential_diff[2])
            new_sol.append(potential_diff[2] + potential_diff[3])
            new_sol.append(potential_diff[1] - potential_diff[3])
            new_sol.sort()
        return new_sol


    def _initial_adjusted_var_from_mean(self, mean):
        return ((self.num_samples - mean * self.num_samples % self.num_samples) *
                (math.floor(mean) - mean) ** 2 +
                (mean * self.num_samples % self.num_samples) *
                (math.ceil(mean) - mean) ** 2
               ) / (self.num_samples - 1) * self.n_minus_1_n_squared


    def _initial_mean_valid_from_mean(self, mean):
        return ([int(math.floor(mean))] *
                (int((self.num_samples - mean * self.num_samples % self.num_samples))) +
                [int(math.ceil(mean))] * 
                int((mean * self.num_samples % self.num_samples))
               )


    def _emit_debug_information_for_valid_means_variances(self, solutions):
        self.simpleData.update(solutions)
        print(str(sum([len(x) for x in self.simpleData.itervalues()])) + " unique solutions found simulatenously (not neccessarily complete!!).")
        index = 0
        for params in self.simpleData:
            if index > 100:
                break
            print("At mean, variance", params, ":")
            for simpleSol in self.simpleData[params]:
                if index > 100:
                    break
                index += 1
                print(simpleSol)
        print("Could not find a unique first solution. Try setting find_first to false. Exiting.")


    def _compute_valid_means_variances(self, findFirst):
        solutions = defaultdict(list)
        means_list = [float(i) / self.num_samples for i in self.mean_range]
        
        mean_variances = []

        step_size = 2 * self.num_samples ** 2
        lower_value = int(math.ceil((self.variance - self.variance_precision) * (self.n_minus_1_n_squared)))
        upper_value = (math.floor((self.variance + self.variance_precision) * (self.n_minus_1_n_squared))) + 1

        # definitely some improvements to make here, perhaps use a generator?
        for m in means_list:
            # construct the minimum variance (not within our range or anything, we are just going to use this for granularity purposes)
            initial_mean_valid = self._initial_mean_valid_from_mean(m)
            initial_adjusted_var = self._initial_adjusted_var_from_mean(m)            

            i = lower_value
            while i < upper_value:
                offset = (i - initial_adjusted_var) % step_size
                if offset == 0:
                    i_over_n_minus_1_n_squared = float(i) / self.n_minus_1_n_squared
                    mean_variances.append((m, i_over_n_minus_1_n_squared))

                    solution = self.adjust_sol(initial_mean_valid, int((i - initial_adjusted_var) / step_size))
                    if solution:
                        solutions[(m, i_over_n_minus_1_n_squared)].append(solution)
                i += step_size - offset

        if self.debug:
            if findFirst and len(solutions) > 0:
                self._emit_debug_information_for_valid_means_variances(solutions)
                return None
        else:
            print("Total potential mean/variance pairs to consider: " + str(len(mean_variances)))
            
        return mean_variances


    def _build_solution_space(self, mean_variance_pairs, check_val=None, poss_vals=None, multiprocess=True):
        if self.debug:
            print("Checking for potential solution spaces.")
        if multiprocess:
            pool = mp.Pool()
            func = functools.partial(multiprocess_get_solution_space,self.min_score, self.max_score, self.num_samples,
                                     check_val=check_val, poss_vals=poss_vals, debug=self.debug)
            solution_spaces = pool.map(func, mean_variance_pairs)
            pool.close()
            pool.join()
        else:
            solution_spaces = [multiprocess_get_solution_space(self.min_score, self.max_score, self.num_samples,
                                                            mean_variance_pair,
                                                            check_val=check_val, poss_vals=poss_vals,
                                                            debug=self.debug)
                               for mean_variance_pair in mean_variance_pairs]
        return solution_spaces


    def _findAll_piece_1_multi_proc(self, solution_spaces, check_val=None, poss_vals=None, multiprocess=True, find_first=False):
        self.sols = {}
        init_base_vecs = []
        init_bases = []
        param_tuples = []
        for solution_space in solution_spaces:
            if solution_space == None:
                continue
            base_vec, basis, _, _, param_tuple = solution_space
            if len(basis) == 0:
                if not any([val<0 for val in base_vec._mat]):
                    sol = base_vec._mat
                    temp_sol =[int(v) for v in sol]
                    if check_val:
                        if isinstance(check_val,int):
                            try:
                                temp_sol[poss_vals.index(check_val)]+=1
                            except ValueError:
                                temp_sol.append(1)
                                poss_vals.append(check_val)
                                # temp_sol[poss_vals.index(val)]=1

                        elif isinstance(check_val, list):
                            for val in check_val:
                                try:
                                    temp_sol[poss_vals.index(val)]+=1
                                except ValueError:
                                    temp_sol.append(1)
                                    poss_vals.append(val)
                                    # temp_sol[poss_vals.index(val)]=1
                        elif isinstance(check_val, dict):
                            for val, num in check_val.iteritems():
                                try:
                                    temp_sol[poss_vals.index(val)]+=1
                                except ValueError:
                                    poss_vals.append(val)
                                    temp_sol.append(1)
                                    # temp_sol[poss_vals.index(val)]=1
                        else:
                            raise TypeError
                    self.sols[param_tuple] = [temp_sol]

                continue
            init_base_vecs.append(base_vec)
            init_bases.append(basis)
            param_tuples.append(param_tuple)
        if self.debug:
            print "Found " + str(len(param_tuples) + len(self.sols)) + " potentially viable mean/variance pairs." #Get Katherine to UPDATE this!
            print "Manipulating Bases and Initial Vectors for Complete Search Guarantee"
        return init_bases, init_base_vecs, param_tuples


    def _findAll_piece_2_multi_proc(self, init_bases, init_base_vecs, param_tuples, check_val=None, poss_vals=None, multiprocess=True, find_first=False):
        pool = mp.Pool()
        bases_and_inits= []
        for X in zip(init_bases, init_base_vecs):
            bases_and_inits.append(multiprocessGetManipBases(X))

        for basis_and_init, param_tuple in zip(bases_and_inits, param_tuples):
            if self.debug:
                print "Checking for solutions at: " + str(param_tuple)
            manip_base_vec = basis_and_init[1]
            manip_basis = basis_and_init[0]
            single_set_sols = multiprocess_recurse_over_solution_path(manip_basis, manip_base_vec)
            temp_sols = []
            for sol in single_set_sols:
                fl_sol = [float(v) for v in sol]
                if not all([x.is_integer() for x in fl_sol]):
                    continue
                temp_sols.append([int(v) for v in fl_sol])
                if check_val:
                    if isinstance(check_val,int):
                        try:
                            temp_sols[-1][poss_vals.index(check_val)]+=1
                        except ValueError:
                            for temp_sol in temp_sols:
                                temp_sol.append(0)
                            poss_vals.append(check_val)
                            temp_sols[-1][poss_vals.index(check_val)]=1

                    elif isinstance(check_val, list):
                        for val in check_val:
                            try:
                                temp_sols[-1][poss_vals.index(val)]+=1
                            except ValueError:
                                for temp_sol in temp_sols:
                                    temp_sol.append(0)
                                poss_vals.append(val)
                                temp_sols[-1][poss_vals.index(val)]=1
                    elif isinstance(check_val, dict):
                        for val, num in check_val.iteritems():
                            try:
                                temp_sols[-1][poss_vals.index(val)]+=1
                            except ValueError:
                                poss_vals.append(val)
                                for temp_sol in temp_sols:
                                    temp_sol.append(0)
                                temp_sols[-1][poss_vals.index(val)]=1
                    else:
                        raise TypeError
            self.sols[param_tuple] = temp_sols

        if self.debug:
            print "Done."
        temp_sols =self.sols
        self.sols = {}
        for key, value in temp_sols.iteritems():
            if len(value)>0:
                self.sols[key] = value
        self.extended_poss_vals = poss_vals
        return self.sols


    def _findFirst_piece_1(self, solution_spaces, check_val=None, poss_vals=None, multiprocess=True, find_first=True):
        base_vecs = []
        bases = []
        if multiprocess:
            init_base_vecs = []
            init_bases = []
            for solution_space in solution_spaces:
                if solution_space == None:
                    continue
                base_vec, basis, _, _, param_tuple = solution_space
                print(basis)
                if len(basis) == 0:
                    if not any([val<0 for val in base_vec._mat]):
                        sol = base_vec._mat
                        temp_sol =[int(v) for v in sol]
                        if check_val:
                            if isinstance(check_val,int):
                                try:
                                    temp_sol[poss_vals.index(check_val)]+=1
                                except ValueError:
                                    temp_sol.append(1)
                                    poss_vals.append(check_val)
                            elif isinstance(check_val, list):
                                for val in check_val:
                                    try:
                                        temp_sol[poss_vals.index(val)]+=1
                                    except ValueError:
                                        temp_sol.append(1)
                                        poss_vals.append(val)
                                        # temp_sol[poss_vals.index(val)]=1
                            elif isinstance(check_val, dict):
                                for val, num in check_val.iteritems():
                                    try:
                                        temp_sol[poss_vals.index(val)]+=1
                                    except ValueError:
                                        poss_vals.append(val)
                                        temp_sol.append(1)
                                        # temp_sol[poss_vals.index(val)]=1
                            else:
                                raise TypeError
                        self.sols[param_tuple] = [temp_sol]
                        self.extended_poss_vals = poss_vals
                        return self.sols
                init_base_vecs.append(base_vec)
                init_bases.append(basis)
            pool = mp.Pool()
            bases_and_inits = pool.map(multiprocessGetManipBases, zip(init_bases, init_base_vecs))
            for basis_and_init in bases_and_inits:
                base_vecs.append(basis_and_init[1])
                bases.append(basis_and_init[0])
            sol = multiprocess_recurse_find_first(bases, base_vecs, covered=set())
        else:
            for solution_space in solution_spaces:
                if solution_space == None:
                    continue
                base_vec, basis, _, _, param_tuple = solution_space
                if len(basis) == 0:
                    if not any([val<0 for val in base_vec._mat]):
                        sol = base_vec._mat
                        temp_sol =[int(v) for v in sol]
                        if check_val:
                            if isinstance(check_val,int):
                                try:
                                    temp_sol[poss_vals.index(check_val)]+=1
                                except ValueError:
                                    temp_sol.append(1)
                                    poss_vals.append(check_val)
                                    # temp_sol[poss_vals.index(val)]=1

                            elif isinstance(check_val, list):
                                for val in check_val:
                                    try:
                                        temp_sol[poss_vals.index(val)]+=1
                                    except ValueError:
                                        temp_sol.append(1)
                                        poss_vals.append(val)
                                        # temp_sol[poss_vals.index(val)]=1
                            elif isinstance(check_val, dict):
                                for val, num in check_val.iteritems():
                                    try:
                                        temp_sol[poss_vals.index(val)]+=1
                                    except ValueError:
                                        poss_vals.append(val)
                                        temp_sol.append(1)
                                        # temp_sol[poss_vals.index(val)]=1
                            else:
                                raise TypeError
                        self.sols[param_tuple] = [temp_sol]
                        self.extended_poss_vals = poss_vals
                        return self.sols
                manip_basis, base_vec = getManipBasis(basis, base_vec)
                manip_base_vec = forced_neg_removal(manip_basis, base_vec)
                base_vecs.append(manip_base_vec)
                bases.append(manip_basis)

            for basis, base_vec in zip(bases, base_vecs):
                sol = recurse_find_first(basis, base_vec)
                if sol:
                    break
        if not sol:
            self.extended_poss_vals = poss_vals
            return None
        sol = [int(v) for v in sol]
        if check_val:
            if isinstance(check_val,int):
                try:
                    sol[poss_vals.index(check_val)]+=1
                except ValueError:
                    sol.append(1)
                    poss_vals.append(check_val)
        
            elif isinstance(check_val, list):
                for val in check_val:
                    try:
                        sol[poss_vals.index(val)]+=1
                    except ValueError:
                        sol.append(1)
                        poss_vals.append(val)
            elif isinstance(check_val, dict):
                for val, num in check_val.iteritems():
                    try:
                        sol[poss_vals.index(val)]+=1
                    except ValueError:
                        poss_vals.append(val)
                        sol.append(1)
            else:
                raise TypeError

        self.sols = {'_':[sol]}
        if self.debug:
            print "Done."
        self.extended_poss_vals = poss_vals
        return self.sols


    def recreateData(self, check_val=None, poss_vals=None, multiprocess=True, find_first=False):
        if poss_vals:
            self.set_ranges_with_possible_values(poss_vals)
        mean_var_pairs = self._compute_valid_means_variances(find_first)

        if not mean_var_pairs:
            print("Could not compute mean-variance pairs. Exiting.")
            return None
            
        solution_spaces = self._build_solution_space(mean_var_pairs, check_val=check_val, poss_vals=poss_vals, multiprocess=multiprocess)

        if find_first:
            # Does not use find_first
            return self._findFirst_piece_1(solution_spaces, check_val=check_val, poss_vals=poss_vals, multiprocess=multiprocess, find_first=find_first)
        else:
            # does not use multiprocess or find_first
            init_bases, init_base_vecs, param_tuples = self._findAll_piece_1_multi_proc(solution_spaces, check_val=check_val, poss_vals=poss_vals, multiprocess=multiprocess, find_first=find_first)
            # does not use multiprocess or find_first
            return self._findAll_piece_2_multi_proc(init_bases, init_base_vecs,param_tuples, check_val=check_val, poss_vals=poss_vals, multiprocess=multiprocess, find_first=find_first)


    def getDataSimple(self):
        if self.simpleData:
            return self.simpleData
        if not self.sols:
            if self.debug:
                print "No solutions to run analysis over."
            raise ValueError
        for param, sol_list in self.sols.iteritems():
            for sol in sol_list:
                simple_sol = []
                poss_vals = self.poss_vals
                if self.extended_poss_vals:
                    poss_vals = self.extended_poss_vals
                for value, num_instances in zip(poss_vals, sol):
                    simple_sol += [value]*num_instances
                self.simpleData[param].append(simple_sol)
        return self.simpleData


if __name__ == "__main__":
    import sys
    RD = RecreateData(1,7,10,3,4, mean_precision=0.5, variance_precision=0.5)
    RD.recreateData(multiprocess=True, find_first=False)
    print RD.getDataSimple()
