from random import choice
import random


class Diffusion:
    def __init__(self, graph_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the list to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.prob_threshold = 0.001

    def getSeedSetProfit(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        ep = 0.0
        for k in range(self.num_product):
            a_n_set = s_total_set.copy()
            a_n_sequence, a_n_sequence2 = [(s, 1) for s in s_set[k]], []
            benefit = self.product_list[k][0]
            product_weight = self.product_weight_list[k]

            while a_n_sequence:
                i_node, i_acc_prob = a_n_sequence.pop(choice([i for i in range(len(a_n_sequence))]))

                # -- notice: prevent the node from owing no receiver --
                if i_node not in self.graph_dict:
                    continue

                i_dict = self.graph_dict[i_node]
                for ii_node in i_dict:
                    if random.random() > i_dict[ii_node]:
                        continue

                    if ii_node in a_n_set:
                        continue
                    a_n_set.add(ii_node)

                    # -- purchasing --
                    ep += benefit * product_weight

                    ii_acc_prob = round(i_acc_prob * i_dict[ii_node], 4)
                    if ii_acc_prob > self.prob_threshold:
                        a_n_sequence2.append((ii_node, ii_acc_prob))

                if not a_n_sequence:
                    a_n_sequence, a_n_sequence2 = a_n_sequence2, a_n_sequence

        return round(ep, 4)

    def getSeedSetProfitBCS(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        ep = 0.0
        for k in range(self.num_product):
            inf_dict = {}
            benefit = self.product_list[k][0]
            product_weight = self.product_weight_list[k]
            for s in [s for s in s_set[k] if s in self.graph_dict]:
                for i in [i for i in self.graph_dict[s] if i not in s_total_set]:
                    if i not in inf_dict:
                        inf_dict[i] = self.graph_dict[s][i]
                    else:
                        inf_dict[i] = round(1.0 - (1.0 - inf_dict[i]) * (1.0 - self.graph_dict[s][i]), 4)

                    if i in self.graph_dict:
                        for j in [j for j in self.graph_dict[i] if j not in s_total_set]:
                            if j not in inf_dict:
                                inf_dict[j] = self.graph_dict[i][j]
                            else:
                                inf_dict[j] = round(1.0 - (1.0 - inf_dict[j]) * (1.0 - self.graph_dict[i][j]), 4)

            ep += sum(inf_dict.values()) * benefit * product_weight

        return round(ep, 4)