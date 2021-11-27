import numpy as np
import heapq


class BostonMechanism(object):
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

        for round in range(self.proposing.shape[1]):
            if verbose:
                print("Round: ", round)

            proposal_this_round = {a: [] for a in range(self.accepting.shape[0])}

            # for each proposer, find the acceptor this round
            # append the proposer to the acceptor's list
            for proposer in range(self.proposing.shape[0]):
                if len(self.p2a_dict[proposer]) > 0:
                    continue
                else:
                    proposal_this_round[self.proposing[proposer][round]].append(
                        proposer
                    )
            self.run_this_round(proposal_this_round)

            # check if all proposers are matched
            stop_mech = True
            for proposer in range(self.proposing.shape[0]):
                if self.p2a_dict[proposer] == []:
                    stop_mech = False
            if stop_mech:
                break

        # loop through the matchings and associete the rankings
        self.find_rankings()

        return self.p2a_dict, self.a2p_dict

    def run_this_round(self, proposal_this_round):
        for acceptor in range(self.accepting.shape[0]):
            # compute how many more proposals this acceptor can take
            accept_remaining_spot = self.acceptor_capacity[acceptor] - len(
                self.a2p_dict[acceptor]
            )
            # if there are no more spots, skip
            if accept_remaining_spot == 0:
                continue
            else:
                accepted_proposers = []
                for proposer in proposal_this_round[acceptor]:
                    acceptor_rank = np.where(self.accepting[acceptor] == proposer)[0][0]
                    accepted_proposers.append((acceptor_rank, proposer))
                accepted_proposers = heapq.nsmallest(
                    accept_remaining_spot, accepted_proposers
                )
                # update acceptor dict
                self.a2p_dict[acceptor].extend(accepted_proposers)

                # update proposer dict
                for proposer in accepted_proposers:
                    self.p2a_dict[proposer[1]].append(acceptor)

    def find_rankings(self):

        for proposer in self.p2a_dict.keys():
            for match in range(len(self.p2a_dict[proposer])):
                ranking = np.where(
                    self.proposing[proposer] == self.p2a_dict[proposer][match]
                )[0][0]
                self.p2a_dict[proposer][match] = (
                    ranking,
                    self.p2a_dict[proposer][match],
                )


if __name__ == "__main__":
    # proposing = np.array([[1, 0, 2], [1, 0, 2], [0, 1, 2]])
    # accepting = np.array([[0, 2, 1], [1, 0, 2], [1, 0, 2]])
    proposing = np.array(
        [
            [1, 5, 0, 4, 3, 2],
            [0, 1, 4, 2, 3, 5],
            [5, 4, 1, 0, 3, 2],
            [1, 0, 2, 5, 3, 4],
            [2, 0, 5, 4, 1, 3],
            [2, 0, 1, 4, 3, 5],
            [1, 0, 2, 4, 5, 3],
            [2, 5, 4, 0, 3, 1],
            [1, 5, 0, 2, 4, 3],
            [1, 3, 5, 0, 2, 4],
        ]
    )

    accepting = np.array(
        [
            [6, 9, 3, 0, 1, 7, 5, 2, 8, 4],
            [6, 0, 8, 3, 9, 7, 5, 4, 1, 2],
            [0, 4, 8, 3, 6, 9, 2, 7, 1, 5],
            [9, 7, 6, 3, 5, 4, 1, 8, 2, 0],
            [2, 1, 9, 0, 7, 8, 4, 3, 6, 5],
            [0, 5, 1, 9, 4, 2, 6, 7, 8, 3],
        ]
    )

    # proposing = np.array([[2, 0, 1], [2, 1, 0], [0, 1, 2], [0, 1, 2], [2, 0, 1]])
    # accepting = np.array([[0, 2, 1, 3, 4], [1, 0, 3, 4, 2], [4, 3, 1, 0, 2]])
    proposer_capacity = np.array([1] * 10)
    acceptor_capacity = np.array([1, 1, 1, 1, 3, 4])
    BM = BostonMechanism(proposing, accepting, proposer_capacity, acceptor_capacity)
    p2a_dict, a2p_dict = BM.run()
    print(p2a_dict)
    print(a2p_dict)
