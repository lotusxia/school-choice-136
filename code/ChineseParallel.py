import numpy as np
import heapq
from DA import DeferredAcceptance


class ChineseParallel(object):
    def __init__(self, proposing, accepting, proposer_capacity, acceptor_capacity):
        """
        proposing: np.arrays of shape (n,k) where there are n proposers 
                   and each proposer has a ranking over k acceptors.
                   Acceptors with smaller index are ranked higher
        accepting: np.arrays of shape (k,n) where there are k acceptors.
                   and each acceptor has a ranking over n proposers.
                   Proposers with smaller index are ranked higher
        proposer_capacity: np.array of shape (n,) 
        acceptor_capacity: np.array of shape (k,)
        """

        self.proposing = proposing.copy()
        self.accepting = accepting.copy()
        self.proposer_capacity = proposer_capacity.copy()
        self.acceptor_capacity = acceptor_capacity.copy()

        self.current_proposing = self.proposing.copy()
        self.current_accepting = self.accepting.copy()
        self.current_proposer_capacity = self.proposer_capacity.copy()
        self.current_acceptor_capacity = self.acceptor_capacity.copy()

        # check validity of preference orders
        self.check_pref_validity()

        # initial proposers and acceptors dict setup
        self.p2a_dict = {}
        self.a2p_dict = {}
        for i in range(proposing.shape[0]):
            self.p2a_dict[i] = []
        for i in range(accepting.shape[0]):
            self.a2p_dict[i] = []

    def check_pref_validity(self):
        # check if preference is valid
        # i.e. no repeated preference
        for proposer in range(self.proposing.shape[0]):
            if len(self.proposing[proposer]) != len(set(self.proposing[proposer])):
                raise ValueError("Proposing preference is not valid")
        for acceptor in range(self.accepting.shape[0]):
            if len(self.accepting[acceptor]) != len(set(self.accepting[acceptor])):
                raise ValueError("Proposing preference is not valid")

    def run(self, breaks, verbose=False):

        # run DA using the first 3 schools
        # print("--------------------------------------CP round 1")
        self.run_this_round(0, breaks[0])
        # print("self.p2a_dict: ", self.p2a_dict)
        # print("--------------------------------------CP round 2")
        self.run_this_round(breaks[0], breaks[1])
        # print("self.p2a_dict: ", self.p2a_dict)
        # print("--------------------------------------CP round 3")
        self.run_this_round(breaks[1], self.proposing.shape[1])
        # print("self.p2a_dict: ", self.p2a_dict)

        self.get_a2p_dict()

        return self.p2a_dict, self.a2p_dict

    def run_this_round(self, starting_pref_index, ending_pref_index):
        """
        RUN DA within the round 
        [starting_pref_index, ending_pref_index)
        """
        # call DA
        DA = DeferredAcceptance(
            self.current_proposing[:, starting_pref_index:ending_pref_index],
            self.current_accepting,
            self.current_proposer_capacity,
            self.current_acceptor_capacity,
        )
        DA_matching, _ = DA.run(verbose=False)
        # print("DA matching: ", DA_matching)

        # get unmatched proposers and acceptors with open slots
        for proposer in DA_matching.keys():
            # if this proposer is matched
            if len(DA_matching[proposer]) > 0:
                # decrement proposer capacity
                self.current_proposer_capacity[proposer] -= 1
                # decrement acceptor capacity
                match = DA_matching[proposer][0][1]
                self.current_acceptor_capacity[match] -= 1

        # update matching dict
        for key in DA_matching.keys():
            if DA_matching[key] != []:
                assert self.p2a_dict[key] == []
                self.p2a_dict[key].append(
                    (
                        DA_matching[key][0][0] + starting_pref_index,
                        DA_matching[key][0][1],
                    )
                )

    def get_a2p_dict(self):
        """
        get a2p dict
        """

        for proposer in self.p2a_dict.keys():
            match = self.p2a_dict[proposer][0][1]
            ranking_at_match = np.where(self.accepting[match] == proposer)[0][0]
            self.a2p_dict[match].append((ranking_at_match, proposer))


if __name__ == "__main__":
    proposing = np.array(
        [
            [6, 1, 0, 4, 5, 3, 2],
            [6, 5, 1, 0, 2, 3, 4],
            [5, 6, 4, 1, 0, 3, 2],
            [1, 0, 2, 5, 3, 4, 6],
            [6, 2, 0, 5, 4, 1, 3],
            [2, 6, 0, 1, 5, 3, 4],
            [6, 1, 5, 2, 4, 0, 3],
            [6, 2, 5, 4, 0, 3, 1],
            [6, 1, 5, 0, 2, 4, 3],
            [1, 3, 5, 6, 0, 2, 4],
        ]
    )

    accepting = np.array(
        [
            [6, 9, 3, 0, 1, 7, 5, 2, 8, 4],
            [6, 0, 8, 3, 9, 7, 5, 4, 1, 2],
            [0, 4, 8, 3, 6, 9, 2, 7, 1, 5],
            [9, 7, 6, 3, 5, 4, 1, 8, 2, 0],
            [2, 1, 9, 0, 7, 8, 4, 3, 6, 5],
            [0, 3, 1, 9, 4, 2, 6, 7, 8, 5],
            [0, 5, 1, 9, 4, 2, 6, 7, 8, 3],
        ]
    )

    # proposing = np.array([[2, 0, 1], [2, 1, 0], [0, 1, 2], [0, 1, 2], [2, 0, 1]])
    # accepting = np.array([[0, 2, 1, 3, 4], [1, 0, 3, 4, 2], [4, 3, 1, 0, 2]])
    proposer_capacity = np.array([1] * 10)
    acceptor_capacity = np.array([1, 1, 2, 1, 3, 4, 1])
    CP = ChineseParallel(proposing, accepting, proposer_capacity, acceptor_capacity)
    p2a_dict = CP.run(breaks=[2, 4], verbose=True)
    print(p2a_dict)

