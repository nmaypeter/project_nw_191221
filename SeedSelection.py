from Initialization import safe_div
from Diffusion import *
import heap


class SeedSelectionMdag:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list, r_flag, epw_flag):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.r_flag = r_flag
        self.epw_flag = epw_flag
        self.prob_threshold = 0.001

    def calculateExpectedProfit(self, s_set):
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        inf_list = [0.0 for _ in range(self.num_product)]

        for k in range(self.num_product):
            inf_dict, inf_dict2 = {s_node: 1.0 for s_node in s_set[k]}, {}
            i_set = set(i_node for s_node in s_set[k] for i_node in self.graph_dict[s_node] if i_node not in inf_dict and i_node not in s_total_set)
            while i_set:
                inf_set = set(i_node for i_node in inf_dict if i_node in self.graph_dict)
                for i_node in i_set:
                    in_set = set(in_node for in_node in inf_set if i_node in self.graph_dict[in_node])
                    i_prod = 0.0
                    for in_node in in_set:
                        i_prod = 1.0 - (1.0 - i_prod) * (1.0 - self.graph_dict[in_node][i_node] * inf_dict[in_node] * (self.product_weight_list[k] if self.epw_flag else 1.0))
                    inf_dict2[i_node] = round(i_prod, 4) if round(i_prod, 4) >= self.prob_threshold else 0.0
                inf_set2 = set(i_node2 for i_node2 in inf_dict2 if i_node2 in self.graph_dict) if sum(inf_dict2.values()) != 0.0 else set()
                i_set = set(i_node for s_node in inf_set2 for i_node in self.graph_dict[s_node] if i_node not in inf_dict and i_node not in inf_dict2 and i_node not in s_total_set)
                inf_dict, inf_dict2 = {**inf_dict, **inf_dict2}, {}
            inf_list[k] = round(sum(inf_dict.values()) - len(s_set[k]), 4)

        ep = round(sum(inf_list[k] * self.product_list[k][0] * (1.0 if self.epw_flag else self.product_weight_list[k])
                       for k in range(self.num_product)), 4)

        return ep

    def generateCelfHeap(self):
        celf_heap = []
        ss = SeedSelectionMdag(self.graph_dict, self.seed_cost_dict, self.product_list, self.product_weight_list, self.r_flag, self.epw_flag)
        for k in range(self.num_product):
            for i in self.graph_dict:
                s_set = [set() for _ in range(self.num_product)]
                s_set[k].add(i)
                ep = ss.calculateExpectedProfit(s_set)

                if ep > 0:
                    if self.r_flag:
                        ep = safe_div(ep, self.seed_cost_dict[i])
                    celf_item = (ep, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

### MIOA, DAG1, DAG2


class SeedSelectionMIOA:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list, dag_class, r_flag, epw_flag):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.dag_class = dag_class
        self.r_flag = r_flag
        self.epw_flag = epw_flag
        self.prob_threshold = 0.001

    def generateMIOA(self):
        ### mioa_dict[source_node][i_node]: (prob., MIP)
        mioa_dict = {}

        for source_node in self.graph_dict:
            ### source_dict: the node in heap which may update its activated probability --
            ### source_dict[i_node] = (prob, in-neighbor)
            mioa_dict[source_node] = {}
            source_dict = {i: (self.graph_dict[source_node][i], source_node) for i in self.graph_dict[source_node]}
            source_dict[source_node] = (1.0, source_node)
            source_heap = [(self.graph_dict[source_node][i], i) for i in self.graph_dict[source_node]]
            heap.heapify_max(source_heap)

            # -- it will not find a better path than the existing MIP --
            # -- because if this path exists, it should be pop earlier from the heap. --
            while source_heap:
                (i_prob, i_node) = heap.heappop_max(source_heap)
                i_prev = source_dict[i_node][1]

                # -- find MIP from source_node to i_node --
                i_path = [i_node, i_prev]
                while i_prev != source_node:
                    i_prev = source_dict[i_prev][1]
                    i_path.append(i_prev)
                i_path.pop()
                i_path.reverse()

                mioa_dict[source_node][i_node] = (i_prob, i_path)

                if i_node in self.graph_dict:
                    for ii_node in self.graph_dict[i_node]:
                        # -- not yet find MIP from source_node to ii_node --
                        if ii_node not in mioa_dict[source_node]:
                            ii_prob = round(i_prob * self.graph_dict[i_node][ii_node], 4)

                            if ii_prob >= self.prob_threshold:
                                # -- if ii_node is in heap --
                                if ii_node in source_dict:
                                    ii_prob_d = source_dict[ii_node][0]
                                    if ii_prob > ii_prob_d:
                                        source_dict[ii_node] = (ii_prob, i_node)
                                        source_heap.remove((ii_prob_d, ii_node))
                                        source_heap.append((ii_prob, ii_node))
                                        heap.heapify_max(source_heap)
                                # -- if ii_node is not in heap --
                                else:
                                    source_dict[ii_node] = (ii_prob, i_node)
                                    heap.heappush_max(source_heap, (ii_prob, ii_node))

        mioa_dict = [mioa_dict] * self.num_product
        if self.epw_flag:
            # -- update node's activated probability by product weight --
            mioa_dict = [{i: {j: (round(mioa_dict[k][i][j][0] * self.product_weight_list[k] ** len(mioa_dict[k][i][j][1]), 4), mioa_dict[k][i][j][1])
                              for j in mioa_dict[k][i]} for i in mioa_dict[k]} for k in range(self.num_product)]
            # -- remove influenced nodes which are over diffusion threshold --
            mioa_dict = [{i: {j: mioa_dict[k][i][j] for j in mioa_dict[k][i] if mioa_dict[k][i][j][0] >= self.prob_threshold} for i in mioa_dict[k]}
                         for k in range(self.num_product)]
            # -- remove empty mioa --
            mioa_dict = [{i: mioa_dict[k][i] for i in mioa_dict[k] if mioa_dict[k][i]} for k in range(self.num_product)]

        return mioa_dict

    def generateDAG1(self, mioa_dict, s_set):
        dag_dict = [{} for _ in range(self.num_product)]
        # -- s_node are influenced by super root --
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        for k in range(self.num_product):
            node_rank_dict = {s_node: 1.0 for s_node in s_set[k]}
            for i in (set(mioa_dict[k]) & s_set[k]):
                for j in mioa_dict[k][i]:
                    i_prod, i_MIP = mioa_dict[k][i][j]
                    if not (set(i_MIP) & s_total_set):
                        if j in node_rank_dict:
                            if i_prod <= node_rank_dict[j]:
                                continue
                        node_rank_dict[j] = i_prod

            # -- i_set collect nodes with out-neighbor --
            i_set = set(i for i in self.graph_dict if i in node_rank_dict)
            for i in i_set:
                # -- j_set collect nodes may be passed information from i --
                j_set = set(j for j in self.graph_dict[i] if j in node_rank_dict and node_rank_dict[i] > node_rank_dict[j])
                dag_dict[k][i] = {j: self.graph_dict[i][j] for j in j_set}
        dag_dict = [{i: dag_dict[k][i] for i in dag_dict[k] if dag_dict[k][i]} for k in range(self.num_product)]

        return dag_dict

    def generateDAG2(self, mioa_dict, s_set):
        dag_dict = [{} for _ in range(self.num_product)]
        s_total_set = set(s for k in range(self.num_product) for s in s_set[k])
        for k in range(self.num_product):
            node_rank_dict = {s_node: 1.0 for s_node in s_total_set}
            i_path_set = set()
            for i in s_set[k]:
                for j in mioa_dict[k][i]:
                    i_prod, i_MIP = mioa_dict[k][i][j]
                    if not (set(i_MIP) & s_total_set):
                        i_path = [i] + i_MIP
                        for len_path in range(len(i_path) - 1):
                            i_path_set.add((i_path[len_path], i_path[len_path + 1]))

                        if j in node_rank_dict:
                            if i_prod <= node_rank_dict[j]:
                                continue
                        node_rank_dict[j] = i_prod
            for i_path in i_path_set:
                (i_node, ii_node) = i_path
                if node_rank_dict[i_node] > node_rank_dict[ii_node]:
                    if i_node not in dag_dict[k]:
                        dag_dict[k][i_node] = {ii_node: self.graph_dict[i_node][ii_node]}
                    else:
                        dag_dict[k][i_node][ii_node] = self.graph_dict[i_node][ii_node]

        dag_dict = [{i: dag_dict[k][i] for i in dag_dict[k] if dag_dict[k][i]} for k in range(self.num_product)]

        return dag_dict

    def calculateExpectedProfit(self, dag_dict, s_set):
        inf_dict = [{s: 1.0 for s in s_set[k]} for k in range(self.num_product)]

        for k in range(self.num_product):
            in_dag_dict = {s: {} for s in s_set[k]}
            for i in dag_dict[k]:
                for j in dag_dict[k][i]:
                    if j not in in_dag_dict:
                        in_dag_dict[j] = {i: dag_dict[k][i][j]}
                    else:
                        in_dag_dict[j][i] = dag_dict[k][i][j]

            root_set = set(j for j in in_dag_dict if not len(in_dag_dict[j]))
            while root_set:
                for i in root_set:
                    del in_dag_dict[i]
                    if i in dag_dict[k]:
                        for j in dag_dict[k][i]:
                            del in_dag_dict[j][i]
                            ii_prob = round(inf_dict[k][i] * dag_dict[k][i][j] * (self.product_weight_list[k] if self.epw_flag else 1.0), 4)
                            if j not in inf_dict[k]:
                                inf_dict[k][j] = ii_prob
                            else:
                                inf_dict[k][j] = 1.0 - (1.0 - inf_dict[k][j]) * (1.0 - ii_prob)
                root_set = set(j for j in in_dag_dict if not len(in_dag_dict[j]))

            for s_node in s_set[k]:
                del inf_dict[k][s_node]

        ep = round(sum(sum(inf_dict[k].values()) * self.product_list[k][0] * (1.0 if self.epw_flag else self.product_weight_list[k])
                       for k in range(self.num_product)), 4)

        return ep

    def generateCelfHeap(self, mioa_dict):
        celf_heap = []
        ss = SeedSelectionMIOA(self.graph_dict, self.seed_cost_dict, self.product_list, self.product_weight_list, self.dag_class, self.r_flag, self.epw_flag)
        for k in range(self.num_product):
            for i in self.graph_dict:
                s_set = [set() for _ in range(self.num_product)]
                s_set[k].add(i)
                dag_dict = [{} for _ in range(self.num_product)]
                if self.dag_class == 1:
                    dag_dict = ss.generateDAG1(mioa_dict, s_set)
                elif self.dag_class == 2:
                    dag_dict = ss.generateDAG2(mioa_dict, s_set)
                ep = ss.calculateExpectedProfit(dag_dict, s_set)

                if ep > 0:
                    if self.r_flag:
                        ep = safe_div(ep, self.seed_cost_dict[i])
                    celf_item = (ep, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

### NG


class SeedSelectionNG:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list, r_flag):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.r_flag = r_flag
        self.monte = 100

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = []

        diff = Diffusion(self.graph_dict, self.product_list, self.product_weight_list)
        for i in self.graph_dict:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = round(sum([diff.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0], self.product_list[0][0] * self.product_weight_list[0])
                    if self.r_flag:
                        mg = safe_div(mg, self.seed_cost_dict[i])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap, celf_item)

        return celf_heap

### HD


class SeedSelectionHD:
    def __init__(self, graph_dict, product_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)

    def generateDegreeHeap(self):
        degree_heap = []

        for i in self.graph_dict:
            deg = len(self.graph_dict[i])
            for k in range(self.num_product):
                degree_item = (int(deg), k, i)
                heap.heappush_max(degree_heap, degree_item)

        return degree_heap

### BCS


def SpiltHeuristicsSet(data_name):
    data_degree_path = 'data/' + data_name + '/degree.txt'
    degree_dict = {}
    with open(data_degree_path) as f:
        for line in f:
            (node, deg) = line.split()
            if int(deg) in degree_dict:
                degree_dict[int(deg)].add(node)
            else:
                degree_dict[int(deg)] = {node}
    f.close()

    degree_count_dict = {deg: len(degree_dict[deg]) for deg in degree_dict}
    num_node20 = sum([degree_count_dict[d] for d in degree_count_dict]) * 0.2
    degree_count_dict = {deg: abs(sum([degree_count_dict[d] for d in degree_count_dict if d >= deg]) - num_node20) for
                         deg in degree_count_dict}
    degree_threshold20 = min(degree_count_dict, key=degree_count_dict.get)
    Billboard_set, Handbill_set = set(), set()
    for deg in degree_dict:
        if deg >= degree_threshold20:
            Billboard_set = Billboard_set.union(degree_dict[deg])
        else:
            Handbill_set = Handbill_set.union(degree_dict[deg])

    return Billboard_set, Handbill_set


class SeedSelectionBalancedCombinationStrategy:
    def __init__(self, graph_dict, seed_cost_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### seed_cost_dict[i]: (float4) the seed of i-node and k-item
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        self.graph_dict = graph_dict
        self.seed_cost_dict = seed_cost_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list

    def generateCelfHeap(self, data_name):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        Billboard_set, Handbill_set = SpiltHeuristicsSet(data_name)
        Billboard_celf_heap, Handbill_celf_heap = [], []

        diff = Diffusion(self.graph_dict, self.product_list, self.product_weight_list)
        for i in Billboard_set:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = diff.getSeedSetProfitBCS(s_set)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0] * self.product_weight_list[k], self.product_list[0][0] * self.product_weight_list[0])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(Billboard_celf_heap, celf_item)

        for i in Handbill_set:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = diff.getSeedSetProfitBCS(s_set)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0], self.product_list[0][0] * self.product_weight_list[0])
                    mg = safe_div(mg, self.seed_cost_dict[i])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(Handbill_celf_heap, celf_item)

        return Billboard_celf_heap, Handbill_celf_heap

### PMIS


class SeedSelectionPMIS:
    def __init__(self, graph_dict, product_list, product_weight_list):
        ### graph_dict: (dict) the graph
        ### product_list: (list) the set to record products [k's profit, k's cost, k's price]
        ### num_product: (int) the kinds of products
        ### product_weight_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = graph_dict
        self.product_list = product_list
        self.num_product = len(product_list)
        self.product_weight_list = product_weight_list
        self.monte = 10

    def generateCelfHeap(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_item: (list) (mg, k_prod, i_node, flag)
        celf_heap = [[] for _ in range(self.num_product)]

        diff_ss = Diffusion(self.graph_dict, self.product_list, self.product_weight_list)
        for i in self.graph_dict:
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = round(sum([diff_ss.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

            if ep > 0:
                for k in range(self.num_product):
                    mg = safe_div(ep * self.product_list[k][0], self.product_list[0][0])
                    celf_item = (mg, k, i, 0)
                    heap.heappush_max(celf_heap[k], celf_item)

        return celf_heap

    def solveMultipleChoiceKnapsackProblem(self, bud, s_matrix, c_matrix):
        mep_result = (0.0, [set() for _ in range(self.num_product)])
        ### bud_index: (list) the using budget index for products
        ### bud_bound_index: (list) the bound budget index for products
        bud_index, bud_bound_index = [len(k) - 1 for k in c_matrix], [0 for _ in range(self.num_product)]
        MCKP_list = []

        diff = Diffusion(self.graph_dict, self.product_list, self.product_weight_list)
        while bud_index != bud_bound_index:
            ### bud_pmis: (float) the budget in this pmis execution
            bud_pmis = sum(c_matrix[k][bud_index[k]] for k in range(self.num_product))

            if bud_pmis <= bud:
                seed_set_flag = True
                if MCKP_list:
                    for senpai_item in MCKP_list:
                        compare_list_flag = True
                        for b_index in bud_index:
                            senpai_index = senpai_item[bud_index.index(b_index)]
                            if b_index > senpai_index:
                                compare_list_flag = False
                                break

                        if compare_list_flag:
                            seed_set_flag = False
                            break

                if seed_set_flag:
                    MCKP_list.append(bud_index.copy())

            pointer = self.num_product - 1
            while bud_index[pointer] == bud_bound_index[pointer]:
                bud_index[pointer] = len(c_matrix[pointer]) - 1
                pointer -= 1
            bud_index[pointer] -= 1

        while MCKP_list:
            bud_index = MCKP_list.pop(0)

            s_set = [s_matrix[k][bud_index[k]][k].copy() for k in range(self.num_product)]
            ep = round(sum([diff.getSeedSetProfit(s_set) for _ in range(self.monte)]) / self.monte, 4)

            if ep > mep_result[0]:
                mep_result = (ep, s_set)

        return mep_result[1]