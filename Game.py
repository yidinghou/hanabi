import random
import numpy as np
from funcs import *

class Card:
    def __init__(self, value, color):
        self.value = value
        self.color = color

    def __str__(self):
        return (str(self.value) + self.color)


class Deck:
    def __init__(self, cards):
        self.cards = cards
        self.values = list(set([c.value for c in cards]))
        self.colors = list(set([c.color for c in cards]))
        self.values.sort()
        self.colors.sort()
        self.empty = False

    def __str__(self):
        return (str([str(c) for c in self.cards]))

    def count(self):
        return len(self.cards)

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, n):
        if self.count() >= n:
            dealt = self.cards[0:n]
            self.cards = self.cards[n:]
            return (dealt)
        else:
            self.empty = True
            return ([])

    def copy(self):
        c = Deck(self.cards)
        return(c)

    def remove(self, c):
        card_str = [str(card) for card in self.cards]
        idx = card_str.index(c)
        ret = self.cards[idx]
        del self.cards[idx]
        return(ret)


class Board:
    def __init__(self, deck):
        d = dict.fromkeys(deck.colors)
        d = dict.fromkeys(d, [0])
        self.colors = colors
        self.slots = d
        self.deck = deck
        self.bombs = 0
        self.clues = 8
        self.score = 0
        self.value = 0
        self.turns = 0
        self.played = []
        self.playable = [x for x in self.deck.cards if x.value == 1]
        self.playableMat = CardProb(deck, bol=True)
        self.correctly_played = []
        self.discardable = []
        self.discardMat = CardProb(deck, bol=True)
        self.finished = False

    def copy(self):
        c = Board(self.colors, self.deck)
        c.slots = self.slots.copy()
        c.deck = self.deck.copy()
        c.bombs = self.bombs
        c.clues = self.clues
        c.score = self.score
        c.played = self.played.copy()
        c.playable = self.playable.copy()
        c.playableMat = self.playableMat.copy()
        c.correctly_played = self.correctly_played.copy()
        c.discardable = self.discardable.copy()
        c.discardMat = self.discardMat.copy()
        c.finished = self.finished

        return (c)

    def play_card(self, card):
        self.played.append(card)
        slot = self.slots[card.color]
        last = slot[-1]
        if card.value == last + 1:
            self.slots[card.color] = slot + [card.value]
            self.score += 1 / 25
            if self.score == 1:
                self.value = 1

            self.discardable.append(card)
            self.playable = [x for x in self.playable if (x.value != card.value) or (x.color != card.color)]
            self.playable += [x for x in self.deck.cards if (x.value == (card.value + 1)) and (x.color == card.color)]

            self.discardMat.possible = self.discardable
            self.discardMat.update(elim_cards=[])

            self.playableMat.possible = self.playable
            self.playableMat.update(elim_cards=[])

            self.correctly_played += [str(card)]
        else:
            self.bombs += 1
            if self.bombs > 2:
                self.finished = True
                # self.score = 0

    def discard(self, card):
        self.turns += 1
        self.played.append(card)

    def remove_clue(self):
        self.clues += -1
        if self.clues < 0:
            self.finished = True

    def add_clue(self):
        self.clues += 1
        self.clues = min(8, self.clues)



class GS():
    def __init__(self, board, deck, curr, opps, action_log, turn):
        self.board = board
        self.curr = curr
        self.opps = opps
        self.turn = turn
        self.deck = deck
        self.nn_input = curr.nn_input(opps[0], board)
        self.id = str(self.nn_input)

        self.log = action_log

    def takeAction(self, action):
        new_done, new_deck, new_curr, new_opps, new_board, new_action = \
            game_act(self.curr.copy(), self.opps[0].copy(), self.deck.copy(), self.board.copy(), action)
        updated_log = self.log + ["%d: %s" % (self.turn + 1, new_action)]
        newState = GS(new_board, new_deck, new_opps[0], [new_curr], updated_log, self.turn + 1)
        return (newState, new_board.score, new_done)


class Node():
    def __init__(self, state):
        self.state = state
        self.id = state.id
        self.edges = []

    def isLeaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge():

    def __init__(self, inNode, outNode, prior, action):
        self.id = inNode.state.id + '|' + outNode.state.id
        self.inNode = inNode
        self.outNode = outNode
        self.action = action

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior, }


class MCTS():
    def __init__(self, root, cpuct):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.addNode(root)

    def __len__(self):
        return len(self.tree)

    def moveToLeaf(self, max_depth = 100):
        breadcrumbs = []
        currentNode = self.root

        done = 0
        value = 0
        simulations = 0
        curr_depth = 0

        while (not currentNode.isLeaf()) and (not done == 1) and (curr_depth<max_depth):
            maxQU = -99999
            curr_depth+=1

            if currentNode == self.root:
                epsilon = 0.2
                nu = np.random.dirichlet([0.8] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)

            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(currentNode.edges):
                simulations += 1
                U = self.cpuct * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(Nb) / (1 + edge.stats['N'])

                Q = edge.stats['Q']

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge

            newState, value, done = currentNode.state.takeAction(
                simulationAction)  # the value of the newState from the POV of the new playerTurn
            if simulations > 1600:
                done = 1
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)
        return currentNode, value, done, breadcrumbs

    def backFill(self, leaf, value, breadcrumbs):
        for edge in breadcrumbs:
            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def addNode(self, node):
        self.tree[node.id] = node