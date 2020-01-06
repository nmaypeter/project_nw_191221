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
            profit_list, time_list = [], []
            for bi in range(10, 6, -1):
                for prod_setting in prod_seq:
                    new_product_name = 'lphc' * (prod_setting == 1) + 'hplc' * (prod_setting == 2)
                    for wd in wd_seq:
                        wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2) + 'm66e34' * (wd == 3)

                        profit, time = [], []
                        profit_dict, time_dict = {model_name: '' for model_name in model_seq}, {model_name: '' for model_name in model_seq}
                        r = new_dataset_name + '\t' + seed_cost_option + '\t' + cascade_model + '\t' + \
                            wallet_distribution_type + '\t' + new_product_name + '\t' + str(bi)
                        print(r)
                        for model_name in model_seq:
                            if model_name == 'mdag1repw' or model_name == 'mdag2repw':
                                best_times, best_pro = -1, -999
                                for times in range(10):
                                    try:
                                        result_name = 'result/' + \
                                                      new_dataset_name + '_' + cascade_model + '_' + seed_cost_option + '/' + \
                                                      wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi) + '/' + \
                                                      model_name + '_' + str(times) + '.txt'

                                        with open(result_name) as f:
                                            p = 0.0
                                            for lnum, line in enumerate(f):
                                                if lnum < 4:
                                                    continue
                                                elif lnum == 4:
                                                    (l) = line.split()
                                                    p = float(l[-1])
                                                elif lnum == 5:
                                                    (l) = line.split()
                                                    c = float(l[-1])
                                                    pro = round(p - c, 4)
                                                    if best_times == -1 or pro >= best_pro:
                                                        best_times = times
                                                        best_pro = pro
                                                else:
                                                    break
                                    except FileNotFoundError:
                                        continue

                                    result_name = 'result/' + \
                                                  new_dataset_name + '_' + cascade_model + '_' + seed_cost_option + '/' + \
                                                  wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi) + '/' + \
                                                  model_name + '_' + str(best_times) + '.txt'

                                    with open(result_name) as f:
                                        p = 0.0
                                        for lnum, line in enumerate(f):
                                            if lnum < 2 or lnum == 3:
                                                continue
                                            elif lnum == 2:
                                                (l) = line.split()
                                                time_dict[model_name] = t
                                            elif lnum == 4:
                                                (l) = line.split()
                                                p = float(l[-1])
                                            elif lnum == 5:
                                                (l) = line.split()
                                                c = float(l[-1])
                                                pro = round(p - c, 4)
                                                profit_dict[model_name] = pro
                                            else:
                                                break
                            else:
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
                                                        time_dict[model_name] = t
                                                    else:
                                                        time_dict[model_name] = round((time_dict[model_name] * times + t) / (times + 1), 4)
                                                elif lnum == 4:
                                                    (l) = line.split()
                                                    p = float(l[-1])
                                                elif lnum == 5:
                                                    (l) = line.split()
                                                    c = float(l[-1])
                                                    pro = round(p - c, 4)
                                                    if times == 0:
                                                        profit_dict[model_name] = pro
                                                    else:
                                                        profit_dict[model_name] = round((profit_dict[model_name] * times + pro) / (times + 1), 4)
                                                else:
                                                    break
                                    except FileNotFoundError:
                                        continue
                            profit.append(str(profit_dict[model_name]))
                            time.append(str(time_dict[model_name]))
                        profit_list.append(profit)
                        time_list.append(time)
                profit_list.append(['' for _ in range(len(model_seq))])
                time_list.append(['' for _ in range(len(model_seq))])

            result_path = 'result/profit_' + new_dataset_name + '.xlsx'
            wb = xw.Book(result_path)
            sheet_name = cascade_model + '_' + seed_cost_option
            sheet = wb.sheets[sheet_name]
            sheet.cells(7, "C").value = profit_list

            result_path = 'result/time_' + new_dataset_name + '.xlsx'
            wb = xw.Book(result_path)
            sheet_name = cascade_model + '_' + seed_cost_option
            sheet = wb.sheets[sheet_name]
            sheet.cells(7, "C").value = time_list