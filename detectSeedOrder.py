from Initialization import *
from SeedSelection import *
import xlwings as xw

data_setting = 1
sc_option = 2
cm = 1
prod_setting = 1
wd = 1
bi = 8

dag_class = 2
r_flag = True
epw_flag = True

model_seq = ['mdag1Mepw', 'mdag2Mepw',
             'mdag1epw', 'mdag1repw', 'mdag2epw', 'mdag2repw',
             'mdag1', 'mdag1pw', 'mdag1r', 'mdag1rpw',
             'mdag2', 'mdag2pw', 'mdag2r', 'mdag2rpw',
             'mng', 'mngpw', 'mngr', 'mngrpw', 'mhd', 'mr']
num_product = 3


dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
               'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
new_dataset_name = 'email' * (data_setting == 1) + 'dnc' * (data_setting == 2) + \
                   'Eu' * (data_setting == 3) + 'Net' * (data_setting == 4)
seed_cost_option = 'dp' * (sc_option == 1) + 'd' * (sc_option == 2) + 'p' * (sc_option == 3)
cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
new_product_name = 'lphc' * (prod_setting == 1) + 'hplc' * (prod_setting == 2)
wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2) + 'm66e34' * (wd == 3)

ini = Initialization(dataset_name, product_name, wallet_distribution_type)
seed_cost_dict = ini.constructSeedCostDict(seed_cost_option)
wallet_dict = ini.constructWalletDict()
num_node = len(wallet_dict)
graph_dict = ini.constructGraphDict(cascade_model)
product_list, product_weight_list = ini.constructProductList()

ssmioa_model = SeedSelectionMIOA(graph_dict, seed_cost_dict, product_list, product_weight_list, dag_class, r_flag, epw_flag)

mioa_dict = ssmioa_model.generateMIOA()
celf_heap = ssmioa_model.generateCelfHeap(mioa_dict)
if r_flag:
    celf_heap = [(safe_div(celf_item[0], seed_cost_dict[celf_item[1]][celf_item[2]]), celf_item[1], celf_item[2], 0) for celf_item in celf_heap]
    heap.heapify_max(celf_heap)

title = ['k', 'i', 'degree', 'wallet', 'cost', 'heap_index']
for model_name in model_seq:
    title.append(model_name)
r_list = [title]
for k in range(num_product):
    for i in seed_cost_dict[k]:
        celf_item = [celf_item for celf_item in celf_heap if celf_item[1] == k and celf_item[2] == i]
        celf_item = celf_item[0] if celf_item else -1
        degree = 0 if i not in graph_dict else len(graph_dict[i])
        celf_item_order = -1 if celf_item == -1 else str(celf_heap.index(celf_item))
        r = [str(k), i, str(degree), str(wallet_dict[i]), str(seed_cost_dict[k][i]), str(celf_item_order)]
        r_list.append(r)

for model_name in model_seq:
    path0 = 'result/' + new_dataset_name + '_' + cascade_model + '_' + seed_cost_option
    path = path0 + '/' + wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi)
    result_name = path + '/' + model_name + '.txt'
    model_dict = [{i: '\t' for i in seed_cost_dict[k]} for k in range(num_product)]

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
        for k in range(num_product):
            for i in seed_cost_dict[k]:
                r_list[1 + k * num_node + int(i)].append(model_dict[k][i])
    except FileNotFoundError:
        for k in range(num_product):
            for i in seed_cost_dict[k]:
                r_list[1 + k * num_node + int(i)].append('')
        continue

result_path = 'analysis/' + new_dataset_name + '_' + cascade_model + '_' + seed_cost_option + '.xlsx'
wb = xw.Book(result_path)
sheet_name = new_product_name + '_' + wallet_distribution_type + '_bi' + str(bi)
sheet = wb.sheets[sheet_name]
sheet.cells(1, "A").value = r_list