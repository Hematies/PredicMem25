import itertools
import math

import numpy as np

from config import Config
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical

class BurstDistributionHandler:

    def __init__(self, config: Config,
                 max_length_kernel_sequences=2,
                 base_sequence_probability=0.5,
                 base_transition_probability=0.2,
                 self_base_transition_probability=0.7,
                 self_sequence_probability=0.7,
                 self_sequence_transition_probability=0.6,
                 self_non_sequence_probability=0.9,
                 self_non_sequence_transition_probability=0.7):
        self.config = config
        self.max_length_kernel_sequences = max_length_kernel_sequences
        self.base_sequence_probability = base_sequence_probability
        self.self_sequence_probability = self_sequence_probability
        self.self_sequence_transition_probability = self_sequence_transition_probability
        self.self_non_sequence_probability = self_non_sequence_probability
        self.self_non_sequence_transition_probability = self_non_sequence_transition_probability
        self.base_transition_probability = base_transition_probability
        self.self_base_transition_probability = self_base_transition_probability

    def __get_deltas(self, addresses):
        deltas = [0]
        deltas.extend([
            addresses[i] - addresses[i - 1]
            for i in range(1, len(addresses))
        ])
        return deltas

    def __get_region_block_address_sequence_mapping(self, addresses):
        region_field_indexing = lambda a: a \
                                          >> (self.config.block_size_log2 + self.config.region_block_size_log2)
        mapping = [(region_field_indexing(a), a >> self.config.block_size_log2)
                   for a in addresses]
        res = {}
        for k, a in mapping:
            if not k in res.keys():
                res[k] = [a]
            else:
                res[k].append(a)
        return res

    def __deltas_to_bursts(self, deltas):
        possible_bursts = [i for i in range(1, self.config.max_block_burst_length + 1)]
        bursts = []
        for delta_ in deltas:
            assert delta_ != 0
            delta = abs(delta_)
            burst = delta if delta in possible_bursts else 1
            bursts.append(burst)
        return bursts

    def __model_sequence_combinations(self, cathegories):
        kernel_sequences = []
        for length in range(2, self.max_length_kernel_sequences+1):
            for kernel in itertools.product(cathegories, repeat=length):
                kernel_sequences.append(list(kernel))
        new_cathegories_map = {i + len(cathegories): kernel_sequences[i] for i in range(0, len(kernel_sequences))}
        return new_cathegories_map

    def __fit_hmm_model(self, block_addresses):
        deltas = self.__get_deltas(block_addresses)
        deltas = [delta for delta in deltas if delta != 0]
        bursts = self.__deltas_to_bursts(deltas)

        if len(bursts) <= 0:
            return None

        cathegories = [burst - 1 for burst in bursts]
        unique, counts = np.unique(cathegories, return_counts=True)
        bins_map = dict(zip(unique, counts))
        cathegories_weight = {cathegory: count / (len(cathegories) / len(list(set(cathegories))))
                              for cathegory, count in bins_map.items()}

        cathegories = sorted(list(set(cathegories)))
        combinations_map = self.__model_sequence_combinations(cathegories)
        cathegories_map = {i: i + 1 if not i in combinations_map
                           else [k + 1 for k in combinations_map[i]]
                           for i in range(len(cathegories) + len(combinations_map))}
        combinations_weights_map = {cathegory:
                                    np.average([cathegories_weight[cathegory_] for cathegory_ in sequence])
                                    for cathegory, sequence in combinations_map.items()}
        all_cathegories = list(cathegories)
        all_cathegories.extend(list(combinations_map.keys()))

        base_weights = [cathegories_weight[cathegory] if not cathegory in combinations_map.keys()
                        else combinations_weights_map[cathegory]
                        for cathegory in all_cathegories]
        sequence_cathegory_weight = self.base_sequence_probability / (len(combinations_map.keys()) / len(all_cathegories))
        base_probabilities = [(1 / len(all_cathegories)) * weight # * sequence_cathegory_weight
                              if cathegory in combinations_map.keys()
                              else (1 / len(all_cathegories)) * weight # / sequence_cathegory_weight
                              for cathegory, weight in zip(all_cathegories, base_weights)]
        base_distribution = Categorical([base_probabilities])
        self_distributions = []

        for i, cathegory in zip(range(0, len(all_cathegories)), all_cathegories):

            sum_other_probabilities = sum([base_probabilities[j] for j in range(0, len(base_probabilities)) if j != i])
            if cathegory in combinations_map.keys():
                scale_correction = (1 - self.self_non_sequence_probability) / sum_other_probabilities
                probabilities = [self.self_non_sequence_probability
                                 if i == j else base_probabilities[j] * scale_correction
                                 for j in range(0, len(base_probabilities))]
            else:
                scale_correction = (1 - self.self_sequence_probability) / sum_other_probabilities
                probabilities = [self.self_sequence_probability
                                 if i == j else base_probabilities[j] * scale_correction
                                 for j in range(0, len(base_probabilities))]
            self_distributions.append(Categorical([probabilities]))
        all_distributions = [base_distribution]
        all_distributions.extend(self_distributions)

        model = DenseHMM(sample_length=1)
        model.add_distributions(all_distributions)
        model.add_edge(base_distribution, base_distribution, self.base_transition_probability)
        model.add_edge(model.start, base_distribution, self.base_transition_probability)
        for i, cathegory0 in zip(range(0, len(all_cathegories)), all_cathegories):
            model.add_edge(model.start, self_distributions[i],
                           (1 - self.base_transition_probability) * base_probabilities[i])

            probability_weight_remainder_in_base_case = 1 - self.self_base_transition_probability
            scale_factor = probability_weight_remainder_in_base_case
            model.add_edge(base_distribution, self_distributions[i], base_probabilities[i] * scale_factor)

            model.add_edge(self_distributions[i], base_distribution, self.base_transition_probability)
            taken_probability_weight = self.base_transition_probability
            if cathegory0 in combinations_map.keys():
                model.add_edge(self_distributions[i], self_distributions[i],
                               self.self_sequence_transition_probability)
                taken_probability_weight = taken_probability_weight + self.self_sequence_transition_probability
            else:
                model.add_edge(self_distributions[i], self_distributions[i],
                               self.self_non_sequence_transition_probability)
                taken_probability_weight = taken_probability_weight + self.self_non_sequence_transition_probability

            probabilities = self_distributions[i].probs[0].tolist()
            sum_rest_probabilities = sum([probabilities[k] for k in range(0, len(probabilities)) if k != i])
            probability_weight_remainder = 1 - taken_probability_weight
            scale_factor = probability_weight_remainder / sum_rest_probabilities

            for j, cathegory1 in zip(range(0, len(all_cathegories)), all_cathegories):
                if i != j:
                    model.add_edge(self_distributions[i], self_distributions[j], probabilities[j] * scale_factor)

        return model, cathegories_map

    def fit_hmm_models_and_generate_bursts_sequences(self, addresses):
        region_field_indexing = lambda a: a \
                                          >> (self.config.block_size_log2 + self.config.region_block_size_log2)
        address_region_list = [(address, region_field_indexing(address)) for address in addresses]
        region_model_table = self.fit_hmm_models(addresses)
        region_sequence_table = {}
        for region, model_and_cathegories_map in region_model_table.items():
            num_accesses = len([address for address, region_ in address_region_list if region == region_])
            if not model_and_cathegories_map is None:
                model, cathegories_map = model_and_cathegories_map
                sequence = self.generate_burst_sequences(model, cathegories_map, num_accesses)
            else:
                sequence = [1 for i in range(0, num_accesses)]
            region_sequence_table[region] = sequence
        bursts, next_bursts = [], []
        for address, region in address_region_list:
            bursts.append(region_sequence_table[region].pop(0))
            if len(region_sequence_table[region]) > 0:
                next_bursts.append(region_sequence_table[region][0])
            else:
                next_bursts.append(bursts[-1])
        return bursts, next_bursts


    def fit_hmm_models(self, addresses):
        mapping = self.__get_region_block_address_sequence_mapping(addresses)
        region_models = {
            region: self.__fit_hmm_model(block_addresses)
            for region, block_addresses in mapping.items()
        }
        return region_models

    def generate_burst_sequences(self, model, cathegories_map, num_accesses):
        cathegories_sequence = model.sample(num_accesses)
        cathegories_sequence = [int(cathegory) for cathegory in cathegories_sequence]
        res = []
        for cathegory in cathegories_sequence:
            value = cathegories_map[cathegory]
            if isinstance(value, list):
                res.extend(value)
            else:
                res.append(value)
        return res[:num_accesses]


