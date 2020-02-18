from skmultiflow.trees.split_criterion import SplitCriterion


class GiniSplitCriterion(SplitCriterion):
    """Misclassification error Impurity split criterion.



    """
    def get_merit_of_split(self, pre_split_dist, post_split_dist):
        total_weight = 0.0
        dist_weights = [0.0]*len(post_split_dist)
        for i in range(len(post_split_dist)):
            dist_weights[i] = sum(post_split_dist[i].values())
            total_weight += dist_weights[i]
        gini = 0.0
        max_weights = 0.0 
        for i in range(len(post_split_dist)):
            gini += (dist_weights[i] / total_weight) * self.compute_gini(post_split_dist[i], dist_weights[i])

        return (1-gini)

    @staticmethod
    def compute_gini(dist, dist_sum_of_weights):
        gini = 1.0
        max_weight = 0.0
        if dist_sum_of_weights != 0.0:
            for _, val in dist.items():
                rel_freq = val / dist_sum_of_weights
                if (rel_freq  > max_weight):
                   max_weight = rel_freq
                #gini -= rel_freq * rel_freq
                #gini -= max(1.0-rel_freq , rel_freq)
        return (1-max_weight)

    @staticmethod
    def get_range_of_merit(pre_split_dist):
        return 1.0
