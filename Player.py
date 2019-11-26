import numpy as np
from Game import *
import config
from funcs import *

class Player():
    def __init__(self, name, deck, model):
        self.deck = deck
        self.cards = deck.cards.copy()
        self.name = name
        self.hand = []
        self.public = []
        self.private = []
        self.public_hand = []
        self.private_hand = []

        self.cpuct = 0.8
        self.MCTSsimulations = 50
        self.model = model
        self.mcts = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def copy(self):
        c = Player(self.name, self.deck.copy(), self.model)
        c.hand = self.hand.copy()
        c.cards = self.cards.copy()
        c.public = self.public.copy()
        c.private = self.private.copy()
        c.public_hand = self.public_hand.copy()
        c.private_hand = self.private_hand.copy()

        c.model = self.model
        c.mtcs = self.mcts

        return (c)

    def print_hand(self):
        print(self.name, [str(c.value) + c.color for c in self.hand])

    def draw(self, cards):
        self.hand += cards
        for c in cards:
            self.public_hand += [CardProb(self.deck)]
            self.private_hand += [CardProb(self.deck)]

    def play_card(self, pos):
        played = self.hand[pos]
        del self.hand[pos]
        del self.public_hand[pos]
        del self.private_hand[pos]

        return played

    def discard_card(self, pos):
        played = self.hand[pos]
        del self.hand[pos]
        del self.public_hand[pos]
        del self.private_hand[pos]

        return played

    def see_players(self, players=[]):
        for p in players:
            self.private += p.hand
        self.private = list(set(self.private))
        for cardMat in self.private_hand:
            cardMat.update(self.private)

    def see_board(self, board):
        self.public += board.played
        self.private += board.played

        self.public = list(set(self.public))
        self.private = list(set(self.private))
        for cardMat in self.private_hand:
            cardMat.update(self.private)

        for cardMat in self.public_hand:
            cardMat.update(self.public)

    def receive_clue(self, clue):
        i = 0
        if type(clue) == str:
            valid = [c for c in self.hand if c.color == clue]
            if not valid == []:
                for card in self.hand:
                    if card.color == clue:
                        elim = [c for c in self.cards if c.color != clue]
                    else:
                        elim = [c for c in self.cards if c.color == clue]
                    self.public_hand[i].update(elim)
                    self.private_hand[i].update(elim)
                    i += 1

        if type(clue) == int:
            valid = [c for c in self.hand if c.value == clue]
            if not valid == []:
                for card in self.hand:
                    if card.value == clue:
                        elim = [c for c in self.cards if c.value != clue]
                    else:
                        elim = [c for c in self.cards if c.value == clue]
                    self.public_hand[i].update(elim)
                    self.private_hand[i].update(elim)
                    i += 1

    def nn_input(self, opp, board):
        nn_public_hand = []
        nn_private_hand = []
        nn_opp_hand = []

        discard_filter = board.discardMat.df.values
        play_filter = board.playableMat.df.values

        for i in range(0, 4):
            public_card = self.public_hand[i].df.values
            private_card = self.private_hand[i].df.values
            opp_pub_card = opp.public_hand[i].df.values
            opp_act_card = card2Mat(opp.hand[i], opp.deck).df.values

            nn_public_hand.append(np.stack((public_card, discard_filter, play_filter)))
            nn_private_hand.append(np.stack((private_card, discard_filter, play_filter)))
            nn_opp_hand.append(np.stack((opp_act_card, opp_pub_card, discard_filter, play_filter)))

        inputs = {"public": nn_public_hand,
                  "private": nn_private_hand,
                  "opp": nn_opp_hand,
                  "board": [board.score, board.bombs, board.clues]}
        return (inputs)

    def simulate(self):
        ##### MOVE THE LEAF NODE
        leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()

        if leaf.id in self.mcts.tree.keys():
            leaf = self.mcts.tree[leaf.id]
        ##### EVALUATE THE LEAF NODE
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)

    def act(self, state, tau):
        if self.mcts == None or state.id not in self.mcts.tree:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)

        #### run the simulation
        for sim in range(self.MCTSsimulations):
            self.simulate()

        #### get action values
        pi, values = self.getAV(1)

        ####pick the action
        action, value = self.chooseAction(pi, values, tau)

        nextState, _, _ = state.takeAction(action)

        NN_value = self.get_preds(nextState)[0]

        return (action, pi, value, NN_value)

    def get_preds(self, state):
        # predict the leaf
        curr = state.curr
        opps = state.opps
        board = state.board

        nn_input = curr.nn_input(opps[0], board)
        probs, value = self.model.predict(np.array([nn_input]))

        return ((probs, value))

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):
        if done == 0:
            probs, value = self.get_preds(leaf.state)
            for action in range(0, 14):
                newState, _, new_done = leaf.state.takeAction(action)
                # if newState.id not in self.mcts.tree.keys():
                #     node = Node(newState)
                #     self.mcts.addNode(node)
                # else:
                #     node = self.mcts.tree[newState.id]
                node = Node(newState)

                if node.id in self.mcts.tree.keys():
                    node = self.mcts.tree[node.id]

                newEdge = Edge(leaf, node, probs[0][action], action)
                leaf.edges.append((action, newEdge))

        return ((value, breadcrumbs))

    def getAV(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(14, dtype=np.integer)
        values = np.zeros(14, dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1 / tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def buildMCTS(self, state):
        self.root = Node(state)
        self.mcts = MCTS(self.root, self.cpuct)

    def changeRootMCTS(self, state):
        self.mcts.root = self.mcts.tree[state.id]

    def replay(self, ltmemory):
        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

            training_states = np.array([row['state'].nn_input for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch])
                , 'policy_head': np.array([row['AV'] for row in minibatch])}

            fit = self.model.fit(training_states, [training_targets["policy_head"], training_targets["value_head"]],
                            epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size=32)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
