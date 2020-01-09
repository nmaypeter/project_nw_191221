from Initialization import *
from SeedSelection import *
import xlwings as xw
import time


if __name__ == '__main__':
    dataset_seq = [1, 2, 3, 4]
    cm_seq = [1, 2]
    prod_seq = [1, 2]
    wd_seq = [1, 2, 3]
    model_seq = ['mdag1Mepw', 'mdag2Mepw',
                 'mdag1epw', 'mdag1repw', 'mdag2epw', 'mdag2repw',
                 'mdag1', 'mdag1pw', 'mdag1r', 'mdag1rpw',
                 'mdag2', 'mdag2pw', 'mdag2r', 'mdag2rpw',
                 'mng', 'mngpw', 'mngr', 'mngrpw', 'mhd', 'mr']
    num_product = 3

    for data_setting in dataset_seq:
        dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                       'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
        new_dataset_name = 'email' * (data_setting == 1) + 'dnc' * (data_setting == 2) + \
                           'Eu' * (data_setting == 3) + 'Net' * (data_setting == 4)
        order_list = []
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            for prod_setting in prod_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
                new_product_name = 'lphc' * (prod_setting == 1) + 'hplc' * (prod_setting == 2)
                for wd in wd_seq:
                    wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2) + 'm66e34' * (wd == 3)
                    bi = 8
                    r_flag = True
                    epw_flag = True
                    for dag_class in [1, 2]:
                        print(new_dataset_name + '\t' + cascade_model + '\t' + new_product_name + '\t' + wallet_distribution_type + '\t' + str(dag_class))
                        now_time = time.time()
                        ini = Initialization(dataset_name, product_name, wallet_distribution_type)
                        seed_cost_dict = ini.constructSeedCostDict()
                        wallet_dict = ini.constructWalletDict()
                        num_node = len(wallet_dict)
                        graph_dict = ini.constructGraphDict(cascade_model)
                        product_list, product_weight_list = ini.constructProductList()

                        ssmioa_model = SeedSelectionMIOA(graph_dict, product_list, product_weight_list, dag_class, r_flag, epw_flag)

                        mioa_dict = ssmioa_model.generateMIOA()
                        celf_heap = ssmioa_model.generateCelfHeap(mioa_dict)
                        if r_flag:
                            celf_heap = [(safe_div(celf_item[0], seed_cost_dict[celf_item[2]]), celf_item[1], celf_item[2], 0)
                                         for celf_item in celf_heap]
                            heap.heapify_max(celf_heap)

                        d = {'data': {}}
                        for k in range(num_product):
                            d['data'][k] = {}
                            for i in seed_cost_dict:
                                celf_item = [celf_item for celf_item in celf_heap if celf_item[1] == k and celf_item[2] == i]
                                celf_item = celf_item[0] if celf_item else -1
                                degree = 0 if i not in graph_dict else len(graph_dict[i])
                                celf_item_order = -1 if celf_item == -1 else celf_heap.index(celf_item)
                                d['data'][k][i] = [str(k), i, str(degree), str(wallet_dict[i]), str(seed_cost_dict[i]), str(celf_item_order)]

                        sort_order = []
                        max_heap_order = 0
                        for model_name in model_seq:
                            d[model_name] = {}
                            path0 = 'resultT/' + new_dataset_name + '_' + cascade_model
                            path = path0 + '/' + wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi)
                            result_name = path + '/' + model_name + '.txt'
                            model_dict = [{i: '' for i in seed_cost_dict} for _ in range(num_product)]

                            try:
                                with open(result_name) as f:
                                    for lnum, line in enumerate(f):
                                        if lnum < 13:
                                            continue
                                        else:
                                            (l) = line.split()
                                            k_prod = l[0]
                                            k_prod = k_prod.replace('(', '')
                                            k_prod = k_prod.replace(',', '')
                                            k_prod = int(k_prod)
                                            i_node = l[1]
                                            i_node = i_node.replace('\'', '')
                                            i_node = i_node.replace(')', '')
                                            model_dict[k_prod][i_node] = str(lnum - 12)
                                            if model_name == 'mdag' + str(dag_class) + 'repw':
                                                sort_order.append((k_prod, i_node))
                                                max_heap_order = max(max_heap_order, int(d['data'][k_prod][i_node][5]))
                                for k in range(num_product):
                                    d[model_name][k] = {}
                                    for i in seed_cost_dict:
                                        d[model_name][k][i] = model_dict[k][i]
                            except FileNotFoundError:
                                for k in range(num_product):
                                    d[model_name][k] = {}
                                    for i in seed_cost_dict:
                                        d[model_name][k][i] = model_dict[k][i]
                                continue
                        order_str = cascade_model + '\t' + new_product_name + '\t' + wallet_distribution_type + '\t' + str(dag_class) + '\t' + str(max_heap_order) + '\n'
                        order_list.append(order_str)

                        r_list = [['k', 'i', 'degree', 'wallet', 'cost', 'heap_index', 'seed_times'] + model_seq]
                        while sort_order:
                            (k, i) = sort_order.pop(0)
                            r = []
                            for model_name in model_seq:
                                r.append(d[model_name][k][i])
                            r = d['data'][k][i] + [str(len([i for i in r if i != '']))] + r
                            r_list.append(r)

                        print(round(time.time() - now_time, 4))
                        result_path = 'analysis/' + new_dataset_name + '_' + cascade_model + '_' + str(dag_class) + '_bi' + str(bi) + '.xlsx'
                        wb = xw.Book(result_path)
                        sheet_name = new_product_name + '_' + wallet_distribution_type
                        sheet = wb.sheets[sheet_name]
                        sheet.cells(1, "A").value = r_list

        fw = open('analysis/' + new_dataset_name + '.txt', 'w')
        for ol_item in order_list:
            fw.write(ol_item)
        fw.close()