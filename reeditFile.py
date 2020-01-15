import os

dataset_seq = [1, 2, 3, 4]
cm_seq = [1, 2]
prod_seq = [1, 2]
wd_seq = [1, 2, 3]
model_seq = ['mdag1epw', 'mdag1repw', 'mdag2epw', 'mdag2repw',
             'mdag1', 'mdag1pw', 'mdag1r', 'mdag1rpw',
             'mdag2', 'mdag2pw', 'mdag2r', 'mdag2rpw',
             'mng', 'mngpw', 'mngr', 'mngrpw', 'mhd', 'mr']

for data_setting in dataset_seq:
    dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                   'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
    new_dataset_name = 'email' * (dataset_name == 'email') + 'dnc' * (dataset_name == 'dnc_email') + \
                       'Eu' * (dataset_name == 'email_Eu_core') + 'Net' * (dataset_name == 'NetHEPT')
    for cm in cm_seq:
        cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        src_path = 'result/' + new_dataset_name + '_' + cascade_model + '_d'
        dst_path = 'result/' + new_dataset_name + '_' + cascade_model
        os.rename(src_path, dst_path)
        for bi in range(10, 6, -1):
            for prod_setting in prod_seq:
                product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
                new_product_name = 'lphc' * (product_name == 'item_lphc') + 'hplc' * (product_name == 'item_hplc')
                for wd in wd_seq:
                    wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2) + 'm66e34' * (wd == 3)
                    r = new_dataset_name + '\t' + cascade_model + '\t' + \
                        wallet_distribution_type + '\t' + new_product_name + '\t' + str(bi)
                    print(r)
                    for model_name in model_seq:
                        for times in range(10):
                            try:
                                result_name = 'result/' + \
                                              new_dataset_name + '_' + cascade_model + '/' + \
                                              wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi) + '/' + \
                                              model_name + '_' + str(times) + '.txt'

                                r = []
                                with open(result_name) as f:
                                    for line in f:
                                        r.append(line)
                                r[0] = new_dataset_name + '_' + cascade_model + '\t' + model_name.split('_')[0] + '\t' + wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi) + '\n'
                                f.close()
                                fw = open(result_name, 'w')
                                for line in r:
                                    fw.write(line)
                                fw.close()

                            except FileNotFoundError:
                                continue