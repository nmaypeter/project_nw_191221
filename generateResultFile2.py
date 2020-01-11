import xlwings as xw

dataset_seq = [1, 2, 3, 4]
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
    for cm in cm_seq:
        cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        index_list, profit_list, time_list = [], [], []
        for bi in range(10, 6, -1):
            for prod_setting in prod_seq:
                new_product_name = 'lphc' * (prod_setting == 1) + 'hplc' * (prod_setting == 2)
                for wd in wd_seq:
                    wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2) + 'm66e34' * (wd == 3)

                    index, profit, time = [], [], []
                    r = new_dataset_name + '\t' + cascade_model + '\t' + \
                        wallet_distribution_type + '\t' + new_product_name + '\t' + str(bi)
                    print(r)
                    for model_name in model_seq:
                        d = {}
                        for times in range(10):
                            try:
                                result_name = 'result/' + \
                                              new_dataset_name + '_' + cascade_model + '/' + \
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
                                            d[times] = pro
                                        else:
                                            break
                            except FileNotFoundError:
                                d[times] = ''

                        if model_name == 'mdag1repw' or model_name == 'mdag2repw' or model_name == 'mdag1Mrepw' or model_name == 'mdag2Mrepw':
                            chosen_index = list(d.keys())[list(d.values()).index(sorted(list(d.values()), reverse=True)[0])]
                        else:
                            if 'epw' in model_name or 'pw' in model_name:
                                chosen_index = list(d.keys())[list(d.values()).index(sorted(list(d.values()), reverse=True)[3])]
                            else:
                                chosen_index = list(d.keys())[list(d.values()).index(sorted(list(d.values()), reverse=True)[5])]

                        try:
                            result_name = 'result/' + \
                                          new_dataset_name + '_' + cascade_model + '/' + \
                                          wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi) + '/' + \
                                          model_name + '_' + str(chosen_index) + '.txt'

                            with open(result_name) as f:
                                index.append(str(chosen_index))
                                p = 0.0
                                for lnum, line in enumerate(f):
                                    if lnum < 2 or lnum == 3:
                                        continue
                                    elif lnum == 2:
                                        (l) = line.split()
                                        t = float(l[-1])
                                        time.append(str(t))
                                    elif lnum == 4:
                                        (l) = line.split()
                                        p = float(l[-1])
                                    elif lnum == 5:
                                        (l) = line.split()
                                        c = float(l[-1])
                                        pro = round(p - c, 4)
                                        profit.append(str(pro))
                                    else:
                                        break
                        except FileNotFoundError:
                            index.append('')
                            time.append('')
                            profit.append('')

                    index_list.append(index)
                    profit_list.append(profit)
                    time_list.append(time)
            index_list.append(['' for _ in range(len(model_seq))])
            profit_list.append(['' for _ in range(len(model_seq))])
            time_list.append(['' for _ in range(len(model_seq))])

        result_path = 'result/index.xlsx'
        wb = xw.Book(result_path)
        sheet_name = new_dataset_name + '_' + cascade_model
        sheet = wb.sheets[sheet_name]
        sheet.cells(7, "C").value = index_list

        result_path = 'result/profit.xlsx'
        wb = xw.Book(result_path)
        sheet_name = new_dataset_name + '_' + cascade_model
        sheet = wb.sheets[sheet_name]
        sheet.cells(7, "C").value = profit_list

        result_path = 'result/time.xlsx'
        wb = xw.Book(result_path)
        sheet_name = new_dataset_name + '_' + cascade_model
        sheet = wb.sheets[sheet_name]
        sheet.cells(7, "C").value = time_list