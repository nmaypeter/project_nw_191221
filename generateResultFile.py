import xlwings as xw

dataset_seq = [1]
sc_option_seq = [2]
cm_seq = [1, 2]
prod_seq = [1, 2]
wd_seq = [1, 3, 2]
model_seq = ['mdag1Mepw', 'mdag2Mepw',
             'mdag1epw', 'mdag1repw', 'mdag2epw', 'mdag2repw',
             'mdag1', 'mdag1pw', 'mdag1r', 'mdag1rpw',
             'mdag2', 'mdag2pw', 'mdag2r', 'mdag2rpw',
             'mng', 'mngpw', 'mngr', 'mngrpw', 'mhd', 'mr']

for data_setting in dataset_seq:
    new_dataset_name = 'email' * (data_setting == 1) + 'dnc' * (data_setting == 2) + \
                    'Eu' * (data_setting == 3) + 'Net' * (data_setting == 4)
    for sc_option in sc_option_seq:
        seed_cost_option = 'dp' * (sc_option == 1) + 'd' * (sc_option == 2) + 'p' * (sc_option == 3)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            profit_max_list, profit_mean_list, profit_min_list, time_max_list, time_mean_list, time_min_list = [], [], [], [], [], []
            for bi in range(10, 6, -1):
                for prod_setting in prod_seq:
                    new_product_name = 'lphc' * (prod_setting == 1) + 'hplc' * (prod_setting == 2)
                    for wd in wd_seq:
                        wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2) + 'm66e34' * (wd == 3)

                        profit_max, profit_mean, profit_min, time_max, time_mean, time_min = [], [], [], [], [], []
                        profit_max_dict, profit_mean_dict, profit_min_dict = {model_name: '' for model_name in model_seq}, {model_name: '' for model_name in model_seq}, {model_name: '' for model_name in model_seq}
                        time_max_dict, time_mean_dict, time_min_dict = {model_name: '' for model_name in model_seq}, {model_name: '' for model_name in model_seq}, {model_name: '' for model_name in model_seq}
                        r = new_dataset_name + '\t' + seed_cost_option + '\t' + cascade_model + '\t' + \
                            wallet_distribution_type + '\t' + new_product_name + '\t' + str(bi)
                        print(r)
                        for model_name in model_seq:
                            for times in range(10):
                                try:
                                    result_name = 'result/' + \
                                                  new_dataset_name + '_' + cascade_model + '_' + seed_cost_option + '/' + \
                                                  wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi) + '/' + \
                                                  model_name + '_' + str(times) + '.txt'

                                    with open(result_name) as f:
                                        p = 0.0
                                        for lnum, line in enumerate(f):
                                            if lnum < 2 or lnum == 3:
                                                continue
                                            elif lnum == 2:
                                                (l) = line.split()
                                                t = float(l[-1])
                                                if times == 0:
                                                    time_max_dict[model_name] = t
                                                    time_mean_dict[model_name] = t
                                                    time_min_dict[model_name] = t
                                                else:
                                                    time_max_dict[model_name] = max(t, time_max_dict[model_name])
                                                    time_mean_dict[model_name] = round((time_mean_dict[model_name] * times + t) / (times + 1), 4)
                                                    time_min_dict[model_name] = min(t, time_max_dict[model_name])
                                            elif lnum == 4:
                                                (l) = line.split()
                                                p = float(l[-1])
                                            elif lnum == 5:
                                                (l) = line.split()
                                                c = float(l[-1])
                                                pro = round(p - c, 4)
                                                if times == 0:
                                                    profit_max_dict[model_name] = pro
                                                    profit_mean_dict[model_name] = pro
                                                    profit_min_dict[model_name] = pro
                                                else:
                                                    profit_max_dict[model_name] = max(pro, profit_max_dict[model_name])
                                                    profit_mean_dict[model_name] = round((profit_mean_dict[model_name] * times + pro) / (times + 1), 4)
                                                    profit_min_dict[model_name] = min(pro, profit_min_dict[model_name])
                                            else:
                                                break
                                except FileNotFoundError:
                                    continue
                            profit_max.append(str(profit_max_dict[model_name]))
                            profit_mean.append(str(profit_mean_dict[model_name]))
                            profit_min.append(str(profit_min_dict[model_name]))
                            time_max.append(str(time_max_dict[model_name]))
                            time_mean.append(str(time_mean_dict[model_name]))
                            time_min.append(str(time_min_dict[model_name]))
                        profit_max_list.append(profit_max)
                        profit_mean_list.append(profit_mean)
                        profit_min_list.append(profit_min)
                        time_max_list.append(time_max)
                        time_mean_list.append(time_mean)
                        time_min_list.append(time_min)
                profit_max_list.append(['' for _ in range(len(model_seq))])
                profit_mean_list.append(['' for _ in range(len(model_seq))])
                profit_min_list.append(['' for _ in range(len(model_seq))])
                time_max_list.append(['' for _ in range(len(model_seq))])
                time_mean_list.append(['' for _ in range(len(model_seq))])
                time_min_list.append(['' for _ in range(len(model_seq))])

            result_path = 'result/profit_max_' + new_dataset_name + '.xlsx'
            wb = xw.Book(result_path)
            sheet_name = cascade_model + '_' + seed_cost_option
            sheet = wb.sheets[sheet_name]
            sheet.cells(7, "C").value = profit_max_list

            result_path = 'result/profit_mean_' + new_dataset_name + '.xlsx'
            wb = xw.Book(result_path)
            sheet_name = cascade_model + '_' + seed_cost_option
            sheet = wb.sheets[sheet_name]
            sheet.cells(7, "C").value = profit_mean_list

            result_path = 'result/profit_min_' + new_dataset_name + '.xlsx'
            wb = xw.Book(result_path)
            sheet_name = cascade_model + '_' + seed_cost_option
            sheet = wb.sheets[sheet_name]
            sheet.cells(7, "C").value = profit_min_list

            result_path = 'result/time_max_' + new_dataset_name + '.xlsx'
            wb = xw.Book(result_path)
            sheet_name = cascade_model + '_' + seed_cost_option
            sheet = wb.sheets[sheet_name]
            sheet.cells(7, "C").value = time_max_list

            result_path = 'result/time_mean_' + new_dataset_name + '.xlsx'
            wb = xw.Book(result_path)
            sheet_name = cascade_model + '_' + seed_cost_option
            sheet = wb.sheets[sheet_name]
            sheet.cells(7, "C").value = time_mean_list

            result_path = 'result/time_min_' + new_dataset_name + '.xlsx'
            wb = xw.Book(result_path)
            sheet_name = cascade_model + '_' + seed_cost_option
            sheet = wb.sheets[sheet_name]
            sheet.cells(7, "C").value = time_min_list