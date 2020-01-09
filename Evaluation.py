from Initialization import *
from random import choice
import random
import time
import os


class Evaluation:
    def __init__(self, graph_dict, prod_list, wallet_dict):
        ### graph_dict: (dict) the graph
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.product_list = prod_list
        self.num_product = len(prod_list)
        self.wallet_dict = wallet_dict

    def getSeedSetProfit(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        seed_diffusion_dict = {(k, s): 0 for k in range(self.num_product) for s in s_set[k]}
        a_n_set = [s_total_set.copy() for _ in range(self.num_product)]
        a_n_sequence, a_n_sequence2 = [(k, s, (k, s)) for k in range(self.num_product) for s in s_set[k]], []
        wallet_dict = self.wallet_dict.copy()

        while a_n_sequence:
            k_prod, i_node, seed_diffusion_flag = a_n_sequence.pop(choice([i for i in range(len(a_n_sequence))]))
            price = self.product_list[k_prod][2]

            for ii_node in self.graph_dict[i_node]:
                if random.random() > self.graph_dict[i_node][ii_node]:
                    continue

                # -- notice: seed cannot use other product --
                if ii_node in a_n_set[k_prod]:
                    continue
                if wallet_dict[ii_node] < price:
                    continue

                # -- purchasing --
                a_n_set[k_prod].add(ii_node)
                wallet_dict[ii_node] -= price
                seed_diffusion_dict[seed_diffusion_flag] += 1

                # -- passing the information --
                if ii_node in self.graph_dict:
                    a_n_sequence2.append((k_prod, ii_node, seed_diffusion_flag))

            if not a_n_sequence:
                a_n_sequence, a_n_sequence2 = a_n_sequence2, a_n_sequence

        pnn_k_list = [len(a_n_set[k]) - len(s_total_set) for k in range(self.num_product)]

        return pnn_k_list, seed_diffusion_dict


class EvaluationM:
    def __init__(self, model_name, dataset_name, product_name, cascade_model):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.new_dataset_name = 'email' * (dataset_name == 'email') + 'dnc' * (dataset_name == 'dnc_email') + \
                                'Eu' * (dataset_name == 'email_Eu_core') + 'Net' * (dataset_name == 'NetHEPT')
        self.product_name = product_name
        self.new_product_name = 'lphc' * (product_name == 'item_lphc') + 'hplc' * (product_name == 'item_hplc')
        self.cascade_model = cascade_model
        self.eva_monte_carlo = 100

    def evaluate(self, bi, wallet_distribution_type, sample_seed_set, ss_time):
        eva_start_time = time.time()
        ini = Initialization(self.dataset_name, self.product_name, wallet_distribution_type)

        seed_cost_dict = ini.constructSeedCostDict()
        graph_dict = ini.constructGraphDict(self.cascade_model)
        product_list = ini.constructProductList()[0]
        num_product = len(product_list)
        wallet_dict = ini.constructWalletDict()
        total_cost = sum(seed_cost_dict[k][i] for i in seed_cost_dict[0] for k in range(num_product))
        total_budget = round(total_cost / 2 ** bi, 4)

        eva = Evaluation(graph_dict, product_list, wallet_dict)
        print('@ evaluation @ ' + self.new_dataset_name + '_' + self.cascade_model +
              '\t' + self.model_name +
              '\t' + wallet_distribution_type + '_' + self.new_product_name + '_bi' + str(bi))
        sample_pnn_k = [0.0 for _ in range(num_product)]
        seed_diffusion_dict_k = {(k, s): 0 for k in range(num_product) for s in sample_seed_set[k]}

        for _ in range(self.eva_monte_carlo):
            pnn_k_list, seed_diffusion_dict = eva.getSeedSetProfit(sample_seed_set)
            sample_pnn_k = [(pnn_k + sample_pnn_k) for pnn_k, sample_pnn_k in zip(pnn_k_list, sample_pnn_k)]
            for seed_diffusion_flag in seed_diffusion_dict:
                seed_diffusion_dict_k[seed_diffusion_flag] += seed_diffusion_dict[seed_diffusion_flag]
        sample_pnn_k = [round(sample_pnn_k / self.eva_monte_carlo, 4) for sample_pnn_k in sample_pnn_k]
        sample_pro_k = [round(sample_pnn_k[k] * product_list[k][0], 4) for k in range(num_product)]
        sample_sn_k = [len(sample_sn_k) for sample_sn_k in sample_seed_set]
        sample_bud_k = [round(sum(seed_cost_dict[k][i] for i in sample_seed_set[k]), 4) for k in range(num_product)]
        sample_bud = round(sum(sample_bud_k), 4)
        sample_pro = round(sum(sample_pro_k), 4)
        seed_diffusion_list = [(seed_diffusion_flag, round(seed_diffusion_dict_k[seed_diffusion_flag] / self.eva_monte_carlo, 4)) for seed_diffusion_flag in seed_diffusion_dict_k]
        seed_diffusion_list = [(round(sd_item[1] * product_list[sd_item[0][0]][0], 4), sd_item[0], sd_item[1]) for sd_item in seed_diffusion_list]
        seed_diffusion_list = sorted(seed_diffusion_list, reverse=True)

        result = [sample_pro, sample_bud, sample_sn_k, sample_pnn_k, sample_pro_k, sample_bud_k, sample_seed_set]
        print('eva_time = ' + str(round(time.time() - eva_start_time, 2)) + 'sec')
        print(result)
        print('------------------------------------------')

        path0 = 'result/' + self.new_dataset_name + '_' + self.cascade_model
        if not os.path.isdir(path0):
            os.mkdir(path0)
        path = path0 + '/' + wallet_distribution_type + '_' + self.new_product_name + '_bi' + str(bi)
        if not os.path.isdir(path):
            os.mkdir(path)
        result_name = path + '/' + self.model_name + '.txt'

        fw = open(result_name, 'w')
        fw.write(self.new_dataset_name + '_' + self.cascade_model + '\t' +
                 self.model_name + '\t' +
                 wallet_distribution_type + '_' + self.new_product_name + '_bi' + str(bi) + '\n' +
                 'budget_limit = ' + str(total_budget) + '\n' +
                 'time = ' + str(ss_time) + '\n\n' +
                 'profit = ' + str(sample_pro) + '\n' +
                 'budget = ' + str(sample_bud) + '\n')
        fw.write('\nprofit_ratio = ')
        for kk in range(num_product):
            fw.write(str(sample_pro_k[kk]) + '\t')
        fw.write('\nbudget_ratio = ')
        for kk in range(num_product):
            fw.write(str(sample_bud_k[kk]) + '\t')
        fw.write('\nseed_number = ')
        for kk in range(num_product):
            fw.write(str(sample_sn_k[kk]) + '\t')
        fw.write('\ncustomer_number = ')
        for kk in range(num_product):
            fw.write(str(sample_pnn_k[kk]) + '\t')
        fw.write('\n\n')

        fw.write(str(sample_seed_set))
        for sd_item in seed_diffusion_list:
            fw.write('\n' + str(sd_item[1]) + '\t' + str(sd_item[0]) + '\t' + str(sd_item[2]))
        fw.close()