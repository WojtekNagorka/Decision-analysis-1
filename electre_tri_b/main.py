from pathlib import Path

import click
import numpy as np
import pandas as pd

from utils import (
    load_dataset,
    load_boundary_profiles,
    load_indifference_thresholds,
    load_preference_thresholds,
    load_veto_thresholds,
    load_criterion_types,
    load_credibility_threshold,
)


# TODO
def calculate_marginal_concordance_index[
    T: (float, np.ndarray),
    U: (float, np.ndarray),
](diff: T, q: U, p: U) -> T:
    """
    Function that calculates the marginal concordance index for the given pair of alternatives, according to the formula presented during classes.

    :param diff: difference between compared alternatives either as a float for single criterion and alternative pairs, or as numpy array for multiple alternatives
    :param q: indifference threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :param p: preference threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :return: marginal concordance index either as a float for single criterion and alternative pairs, or as numpy array for multiple criterion
    """
    if (diff >= p):
        return 0
    elif (diff <= q):
        return 1
    else:
        return (p - diff)/(p - q)

# TODO
def calculate_marginal_concordance_matrix(
    dataset: pd.DataFrame,
    boundary_profiles: pd.DataFrame,
    indifference_thresholds,
    preference_thresholds,
    criterion_types,
) -> np.ndarray:
    """
    Function that calculates the marginal concordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param boundary_profiles: pandas dataframe with boundary profiles
    :param indifference_thresholds: pandas dataframe representing indifference thresholds for all boundary profiles and criterion
    :param preference_thresholds: pandas dataframe representing preference thresholds for all boundary profiles and criterion
    :param criterion_types: pandas dataframe with a column 'type' representing the type of criterion (either gain or cost)
    :return: 4D numpy array with marginal concordance matrix with shape [2, number of alternatives, number of boundary profiles, number of criterion], where element with index [0, i, j, k] describe marginal concordance index between alternative i and boundary profile j on criterion k, while element with index [1, i, j, k] describe marginal concordance index between boundary profile j and  alternative i on criterion k
    """
    concordance_matrix = np.zeros((2, len(dataset), len(boundary_profiles), len(criterion_types)))
    for row1_id, row in dataset.iterrows():
        for row2_id, row2 in boundary_profiles.iterrows():
            for ix in len(row):
                cr_name = dataset.columns[ix]
                cr_type = criterion_types["type"][ix]
                q = indifference_thresholds[cr_name][row2_id]
                p = preference_thresholds[cr_name][row2_id]

                # a P b
                if cr_type == 'gain':
                    diff = row2[cr_name] - row[cr_name]
                else:
                    diff = row[cr_name] - row2[cr_name]
                
                conc_ind = calculate_marginal_concordance_index(diff, q, p)
                concordance_matrix[0, row1_id, row2_id, ix] = conc_ind

                # b P a
                if cr_type == 'gain':
                    diff = row[cr_name] - row2[cr_name]
                else:
                    diff = row2[cr_name] - row[cr_name]

                conc_ind = calculate_marginal_concordance_index(diff, q, p)
                concordance_matrix[1, row1_id, row2_id, ix] = conc_ind
    return concordance_matrix

# TODO
def calculate_comprehensive_concordance_matrix(
    marginal_concordance_matrix: np.ndarray, criterion_types: pd.DataFrame
) -> np.ndarray:
    """
    Function that calculates comprehensive concordance matrix for the given dataset

    :param marginal_concordance_matrix: 4D numpy array with marginal concordance matrix with shape [2, number of alternatives, number of boundary profiles, number of criterion], where element with index [0, i, j, k] describe marginal concordance index between alternative i and boundary profile j on criterion k, while element with index [1, i, j, k] describe marginal concordance index between boundary profile j and  alternative i on criterion k
    :param criterion_types: dataframe that contains "k" column with criterion weights
    :return: 3D numpy array with comprehensive concordance matrix with shape [2, number of alternatives, number of boundary profiles], where element with index [0, i, j] describe comprehensive concordance index between alternative i and boundary profile j, while element with index [1, i, j] describe comprehensive concordance index between boundary profile j and  alternative i
    """
    weights = criterion_types['k']
    marginal_concordance_matrix = marginal_concordance_matrix * weights
    return np.sum(marginal_concordance_matrix, axis = 3)

# TODO
def calculate_marginal_discordance_index[
    T: (float, np.ndarray),
    U: (float, np.ndarray),
](diff: T, p: U, v: U) -> T:
    """
    Function that calculates the marginal concordance index for the given pair of alternatives, according to the formula presented during classes.

    :param diff: difference between compared alternatives either as a float for single criterion and alternative pairs, or as numpy array for multiple alternatives
    :param p: preference threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :param v: veto threshold either as a float if you prefer to calculate for a single criterion or as numpy array for multiple criterion
    :return: marginal discordance index either as a float for single criterion and alternative pairs, or as numpy array for multiple criterion
    """
    if (diff <= p):
        return 0
    elif (diff >= v):
        return 1
    else:
        return (diff - p)/(v - p)

# TODO
def calculate_marginal_discordance_matrix(
    dataset: pd.DataFrame,
    boundary_profiles: pd.DataFrame,
    preference_thresholds,
    veto_thresholds,
    criterion_types,
) -> np.ndarray:
    """
    Function that calculates the marginal discordance matrix for all alternatives pairs and criterion available in dataset

    :param dataset: pandas dataframe representing dataset with alternatives as rows and criterion as columns
    :param boundary_profiles: pandas dataframe with boundary profiles
    :param preference_thresholds: pandas dataframe representing preference thresholds for all boundary profiles and criterion
    :param veto_thresholds: pandas dataframe representing veto thresholds for all boundary profiles and criterion
    :param criterion_types: pandas dataframe with a column 'type' representing the type of criterion (either gain or cost)
    :return: 4D numpy array with marginal discordance matrix with shape [2, number of alternatives, number of boundary profiles, number of criterion], where element with index [0, i, j, k] describe marginal discordance index between alternative i and boundary profile j on criterion k, while element with index [1, i, j, k] describe marginal discordance index between boundary profile j and  alternative i on criterion k
    """
    discordance_matrix = np.zeros((2, len(dataset), len(boundary_profiles), len(criterion_types)))
    for row1_id, row in dataset.iterrows():
        for row2_id, row2 in boundary_profiles.iterrows():
            for ix in len(row):
                cr_name = dataset.columns[ix]
                cr_type = criterion_types["type"][ix]
                v = veto_thresholds[cr_name][row2_id]
                p = preference_thresholds[cr_name][row2_id]

                # a P b
                if cr_type == 'gain':
                    diff = row2[cr_name] - row[cr_name]
                else:
                    diff = row[cr_name] - row2[cr_name]
                
                disc_ind = calculate_marginal_discordance_index(diff, p, v)
                discordance_matrix[0, row1_id, row2_id, ix] = disc_ind

                # b P a
                if cr_type == 'gain':
                    diff = row[cr_name] - row2[cr_name]
                else:
                    diff = row2[cr_name] - row[cr_name]

                disc_ind = calculate_marginal_discordance_index(diff, p, v)
                discordance_matrix[1, row1_id, row2_id, ix] = disc_ind
    return discordance_matrix

# TODO
def calculate_credibility_index(
    comprehensive_concordance_matrix: np.ndarray,
    marginal_discordance_matrix: np.ndarray,
) -> np.ndarray:
    """
    Function that calculates the credibility index for the given comprehensive concordance matrix and marginal discordance matrix

    :param comprehensive_concordance_matrix: 3D numpy array with comprehensive concordance matrix. Every entry in the matrix [i, j] represents comprehensive concordance index between alternative i and alternative j
    :param marginal_discordance_matrix: 4D numpy array with marginal discordance matrix, Consecutive indices [i, j, k] describe first alternative, second alternative, criterion
    :return: 3D numpy array with credibility matrix with shape [2, number of alternatives, number of boundary profiles], where element with index [0, i, j] describe credibility index between alternative i and boundary profile j, while element with index [1, i, j] describe credibility index between boundary profile j and  alternative i
    """
    credibility_matrix = np.zeros_like(comprehensive_concordance_matrix)
    for i in range(2):
        for j in range(credibility_matrix.shape[1]):
            for k in range(credibility_matrix.shape[2]):
                conc_idx = comprehensive_concordance_matrix[i, j, k]

                if conc_idx == 1:
                    credibility_matrix[i, j, k] = 1
                    continue
                
                credibility = conc_idx
                for l in range(marginal_discordance_matrix.shape[3]):
                    disc_idx = marginal_discordance_matrix[i, j, k, l]
                    if disc_idx <= conc_idx:
                        continue
                    credibility *= (1 - disc_idx)/(1 - conc_idx)
                credibility_matrix[i, j, k] = credibility
    return credibility_matrix

# TODO
def calculate_outranking_relation_matrix(
    credibility_index: np.ndarray, credibility_threshold
) -> np.ndarray:
    """
    Function that calculates boolean matrix with information if outranking holds for a given pair

    :param credibility_index: 3D numpy array with credibility matrix with shape [2, number of alternatives, number of boundary profiles], where element with index [0, i, j] describe credibility index between alternative i and boundary profile j, while element with index [1, i, j] describe credibility index between boundary profile j and  alternative i
    :param credibility_threshold: float number
    :return: 3D numpy boolean matrix with information if outranking holds for a given pair
    """
    return credibility_index > credibility_threshold

# TODO
def calculate_relation(
    outranking_relation_matrix: np.ndarray,
    alternatives: pd.Index,
    boundary_profiles_names: pd.Index,
) -> pd.DataFrame:
    """
    Function that determine relation between alternatives and boundary profiles

    :param outranking_relation_matrix: 3D numpy boolean matrix with information if outranking holds for a given pair
    :param alternatives: names of alternatives
    :param boundary_profiles_names: names of boundary profiles
    :return: pandas dataframe with relation between alternatives as rows and boundary profiles as columns. Use "<" or ">" for preference, "I" for indifference and "?" for incompatibility
    """
    relation_matrix = np.zeros_like((len(alternatives), len(boundary_profiles_names)))
    for i in range(len(alternatives)):
        for j in range(len(boundary_profiles_names)):
            if outranking_relation_matrix[0, i, j] == True and outranking_relation_matrix[1, i, j] == True:
                rel = 'I'
            elif outranking_relation_matrix[0, i, j] == True and outranking_relation_matrix[1, i, j] == False:
                rel = '>'
            elif outranking_relation_matrix[0, i, j] == False and outranking_relation_matrix[1, i, j] == True:
                rel = '<'
            else:
                rel = '?'
            relation_matrix[i, j] = rel
    return pd.DataFrame(relation_matrix, columns=boundary_profiles_names, index=alternatives)

# TODO
def calculate_pessimistic_assigment(relation: pd.DataFrame) -> pd.DataFrame:
    """
    Function that calculates pessimistic assigment for given relation between alternatives and boundary profiles

    :param relation: pandas dataframe with relation between alternatives as rows and boundary profiles as columns. With "<" or ">" for preference, "I" for indifference and "?" for incompatibility
    :return: dataframe with pessimistic assigment
    """
    assignment = np.zeros((len(relation.index)))
    for idx, row in relation.iterrows():
        prev = '>'
        for i in range(len(row.values)):
            if prev == '>' and not row.values[i] =='>':
                assignment[idx] = relation.columns[i]
                break
            else:
                prev = row.values[i]
    return assignment

# TODO
def calculate_optimistic_assigment(relation: pd.DataFrame) -> pd.DataFrame:
    """
    Function that calculates optimistic assigment for given relation between alternatives and boundary profiles

    :param relation: pandas dataframe with relation between alternatives as rows and boundary profiles as columns. With "<" or ">" for preference, "I" for indifference and "?" for incompatibility
    :return: dataframe with optimistic assigment
    """
    assignment = np.zeros((len(relation.index)))
    for idx, row in relation.iterrows():
        prev = '>'
        for i in range(len(row.values)):
            if prev == '>' and row.values[i] =='<':
                assignment[idx] = relation.columns[i-1]
                break
            else:
                prev = row.values[i]
    return assignment


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def promethee(dataset_path: Path) -> None:
    dataset_path = Path(dataset_path)

    dataset = load_dataset(dataset_path)
    boundary_profiles = load_boundary_profiles(dataset_path)
    criterion_types = load_criterion_types(dataset_path)
    indifference_thresholds = load_indifference_thresholds(dataset_path)
    preference_thresholds = load_preference_thresholds(dataset_path)
    veto_thresholds = load_veto_thresholds(dataset_path)
    credibility_threshold = load_credibility_threshold(dataset_path)

    marginal_concordance_matrix = calculate_marginal_concordance_matrix(
        dataset,
        boundary_profiles,
        indifference_thresholds,
        preference_thresholds,
        criterion_types,
    )
    comprehensive_concordance_matrix = calculate_comprehensive_concordance_matrix(
        marginal_concordance_matrix, criterion_types
    )

    marginal_discordance_matrix = calculate_marginal_discordance_matrix(
        dataset,
        boundary_profiles,
        preference_thresholds,
        veto_thresholds,
        criterion_types,
    )

    credibility_index = calculate_credibility_index(
        comprehensive_concordance_matrix, marginal_discordance_matrix
    )
    outranking_relation_matrix = calculate_outranking_relation_matrix(
        credibility_index, credibility_threshold
    )
    relation = calculate_relation(
        outranking_relation_matrix, dataset.index, boundary_profiles.index
    )

    pessimistic_assigment = calculate_pessimistic_assigment(relation)
    optimistic_assigment = calculate_optimistic_assigment(relation)

    print("pessimistic assigment\n", pessimistic_assigment)
    print("optimistic assigment\n", optimistic_assigment)


if __name__ == "__main__":
    promethee()
