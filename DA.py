import numpy as np
import heapq


class DeferredAcceptance(object):
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

        self.proposing = proposing
        self.accepting = accepting
        self.proposer_capacity = proposer_capacity
        self.acceptor_capacity = acceptor_capacity

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

    def run(self, verbose=False):
        """
        Take the proposing np.array and the accepting np.array and 
        return the matching between proposers and acceptors.

        Returns:
            two dictionaries, one for proposers and one for acceptors.
        """
        # keep track of the acceptors who have not been proposed
        left_to_propose = self.proposing.copy()

        round = 0

        while self.keep_going(left_to_propose):

            if verbose:
                print("Round", round)
            round += 1

            # for each proposer, propose the the favorate remaining acceptor
            for proposer in range(self.proposing.shape[0]):

                # if proposer is not at capacity
                if len(self.p2a_dict[proposer]) < self.proposer_capacity[proposer]:

                    # check if this proposer has a new potential match
                    if (left_to_propose[proposer] < 0).all():
                        continue
                    else:
                        # propose to the first remaining acceptor(s)
                        remaining_spots = self.proposer_capacity[proposer] - len(
                            self.p2a_dict[proposer]
                        )
                        remaining_acceptors = (left_to_propose[proposer] >= 0).sum()

                        # propose to min(remaining_spots, remaining_acceptors)
                        for sp in range(min(remaining_spots, remaining_acceptors)):
                            starting_pos = np.where(left_to_propose[proposer] >= 0)[0][
                                0
                            ]
                            acceptor = left_to_propose[proposer][starting_pos]
                            # print("\t", proposer, "-->", acceptor)
                            self.run_this_pair(starting_pos, proposer, acceptor)
                            left_to_propose[proposer][starting_pos] = -99

        # before returning, negate preference orders
        for proposer in self.p2a_dict.keys():
            for match_idx in range(len(self.p2a_dict[proposer])):
                self.p2a_dict[proposer][match_idx] = (
                    -self.p2a_dict[proposer][match_idx][0],
                    self.p2a_dict[proposer][match_idx][1],
                )
        for acceptor in self.a2p_dict.keys():
            for match_idx in range(len(self.a2p_dict[acceptor])):
                self.a2p_dict[acceptor][match_idx] = (
                    -self.a2p_dict[acceptor][match_idx][0],
                    self.a2p_dict[acceptor][match_idx][1],
                )

        return self.p2a_dict, self.a2p_dict

    def keep_going(self, left_to_propose):
        """
        Called in run() to determine if the algorithm should continue.

        Algorithm stops when all proposers have been matched.
        """
        keep_going = False
        for proposer in range(self.proposing.shape[0]):
            # if there're acceptors who have not rejected this proposer
            more_option = (left_to_propose[proposer] >= 0).any()
            # if proposer is not at capacity
            not_filled = len(self.p2a_dict[proposer]) < self.proposer_capacity[proposer]
            if not_filled and more_option:
                keep_going = True
                break
        return keep_going

    def run_this_pair(self, proposer_rank, proposer, acceptor):

        rejected_proposer = None

        # update acceptor dict
        accept_rank = np.where(self.accepting[acceptor] == proposer)[0][0]
        heapq.heappush(self.a2p_dict[acceptor], (-accept_rank, proposer))
        if len(self.a2p_dict[acceptor]) > self.acceptor_capacity[acceptor]:
            rejected_proposer = heapq.heappop(self.a2p_dict[acceptor])[1]

        # update proposer dict
        if rejected_proposer is None:
            # print("\t", proposer, "-->", acceptor, "(accepted)")
            heapq.heappush(self.p2a_dict[proposer], (-proposer_rank, acceptor))

        else:
            if rejected_proposer != proposer:
                # print(
                #     "\t",
                #     proposer,
                #     "-->",
                #     acceptor,
                #     "(accepted)",
                #     "but",
                #     rejected_proposer,
                #     "rejected",
                # )
                # update accepted proposer's match
                heapq.heappush(self.p2a_dict[proposer], (-proposer_rank, acceptor))

                # update rejector proposer's match
                for i in range(len(self.p2a_dict[rejected_proposer])):
                    if self.p2a_dict[rejected_proposer][i][1] == acceptor:
                        break
                self.p2a_dict[rejected_proposer].pop(i)
            # else:
            # print("\t", proposer, "-->", acceptor, "(rejected)")


if __name__ == "__main__":
    proposing = np.array([[2, 0, 1], [2, 1, 0], [0, 1, 2], [0, 1, 2], [0, 2, 1]])
    accepting = np.array([[0, 2, 1, 3, 4], [1, 0, 3, 4, 2], [4, 3, 1, 0, 2]])
    # accepting = np.array([[1, 0, 2], [0, 2, 1], [0, 1, 2]])
    # proposing = np.array([[0, 2, 1], [2, 0, 1], [0, 2, 1]])
    proposer_capacity = np.array([1, 1, 1, 1, 1])
    acceptor_capacity = np.array([1, 1, 1])
    DA = DeferredAcceptance(proposing, accepting, proposer_capacity, acceptor_capacity)
    p2a_dict, a2p_dict = DA.run(verbose=True)
    print(p2a_dict)
    print(a2p_dict)
