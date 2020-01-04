from generateDistribution import *


def safe_div(x, y):
    if y == 0:
        return 0.0
    return round(x / y, 4)


class Initialization:
    def __init__(self, data_name, prod_name, wallet_dist_type):
        self.data_name = data_name
        self.data_ic_weight_path = 'data/' + data_name + '/weight_ic.txt'
        self.data_wc_weight_path = 'data/' + data_name + '/weight_wc.txt'
        self.data_degree_path = 'data/' + data_name + '/degree.txt'
        self.prod_name = prod_name
        self.product_path = 'item/' + prod_name + '.txt'
        self.wallet_dist_type = wallet_dist_type
        self.wallet_dict_path = 'data/' + data_name + '/wallet_' + prod_name.split('_')[1] + '_' + wallet_dist_type + '.txt'

    def constructSeedCostDict(self, sc_option):
        # -- calculate the cost for each seed --
        ### seed_cost_dict[k][i]: (float4) the seed of i-node and k-item
        prod_cost_list = []
        max_price = 0.0
        with open(self.product_path) as f:
            for line in f:
                (b, c, r, p) = line.split()
                prod_cost_list.append(float(c))
                max_price = max(max_price, float(p))
        prod_cost_list = [round(prod_cost / max_price, 4) for prod_cost in prod_cost_list]
        num_product = len(prod_cost_list)

        seed_cost_dict, deg_dict = [{} for _ in range(num_product)], {}
        max_deg = 0
        with open(self.data_degree_path) as f:
            for line in f:
                (node, degree) = line.split()
                max_deg = max(max_deg, int(degree))
                deg_dict[node] = int(degree)
        f.close()

        alpha = 0.5
        if sc_option == 'd':
            alpha = 1
        elif sc_option == 'p':
            alpha = 0

        for k in range(num_product):
            item_cost = prod_cost_list[k]
            for i in deg_dict:
                seed_cost_dict[k][i] = round(alpha * safe_div(deg_dict[i], max_deg) + (1 - alpha) * item_cost, 4)

        return seed_cost_dict

    def constructGraphDict(self, cas):
        # -- build graph --
        ### graph: (dict) the graph
        ### graph[node1]: (dict) the set of node1's receivers
        ### graph[node1][node2]: (float) the weight one the edge of node1 to node2
        path = self.data_ic_weight_path * (cas == 'ic') + self.data_wc_weight_path * (cas == 'wc')
        graph = {}
        with open(path) as f:
            for line in f:
                (node1, node2, wei) = line.split()
                if node1 in graph:
                    graph[node1][node2] = float(wei)
                else:
                    graph[node1] = {node2: float(wei)}
        f.close()

        return graph

    def constructProductList(self):
        # -- get product list --
        ### prod_list: (list) [profit, cost, price]
        prod_list, price_list = [], []
        with open(self.product_path) as f:
            for line in f:
                (b, c, r, p) = line.split()
                prod_list.append([float(b), float(c), float(p)])
                price_list.append(float(p))
        f.close()

        max_price = max([price for price in price_list])
        prod_list = [[round(detail / max_price, 4) for detail in item] for item in prod_list]

        pw_list = [1.0 for _ in range(len(price_list))]
        if self.wallet_dist_type in ['m50e25', 'm99e96', 'm66e34']:
            mu, sigma = 0, 1
            if self.wallet_dist_type == 'm50e25':
                mu = np.mean(price_list)
                sigma = (max(price_list) - mu) / 0.6745
            elif self.wallet_dist_type == 'm99e96':
                mu = sum(price_list)
                sigma = abs(min(price_list) - mu) / 2.915
            elif self.wallet_dist_type == 'm66e34':
                mu = sum(price_list) * 0.4167
                sigma = abs(max(price_list) - mu) / 0.4125
            X = np.arange(0, 2, 0.001)
            Y = stats.norm.sf(X, mu, sigma)
            pw_list = [round(float(Y[np.argwhere(X == p)]), 4) for p in price_list]

        return prod_list, pw_list

    def constructWalletDict(self):
        # -- get wallet_list from file --
        wallet_dict = {}
        with open(self.wallet_dict_path) as f:
            for line in f:
                (node, wal) = line.split()
                wallet_dict[node] = float(wal)
        f.close()

        return wallet_dict