from memory import Memory
import config
from Game import *
import pandas as pd
from collections import Counter

colors = ["B","G","R","W","Y"]
numbers = [1,1,1,2,2,3,3,4,4,5]

class CardProb():
    def __init__(self, deck, bol=False):
        self.df = pd.DataFrame(np.zeros((len(deck.values), len(deck.colors))),
                               columns=deck.colors,
                               index=deck.values)
        self.deck = deck
        self.possible = deck.cards
        self.bool = bol
        self.update(elim_cards=[])

    def update(self, elim_cards):
        self.possible = [x for x in self.possible if x not in elim_cards]
        self.df = pd.DataFrame(np.zeros((len(self.deck.values), len(self.deck.colors))),
                               columns=self.deck.colors,
                               index=self.deck.values)
        freq = dict(Counter([str(c) for c in self.possible]))
        for k in freq.keys():
            idx = int(k[0])
            col = k[1]
            self.df.loc[idx, col] = freq[k]
        count = sum(sum(self.df.values))
        self.df = self.df / count

        if self.bool == True:
            self.df = (self.df > 0) * 1


def card2Mat(card, deck):
    mat = CardProb(deck)
    mat.possible = [card]
    mat.update(elim_cards=[])
    return(mat)


def game_act(curr, opp, deck, board, act_num):
    done = 0
    curr.see_players([opp])
    curr.see_board(board)
    action_log = ""

    if act_num < 5:
        discard = curr.play_card(act_num)
        board.discard(discard)
        curr.draw(deck.deal(1))
        board.add_clue()
        action_log = "Action %s discarded: %s" % (curr.name, discard)
    elif act_num < 10:
        play = curr.play_card(act_num - 5)
        curr.draw(deck.deal(1))
        board.play_card(play)
        action_log = "Action %s played: %s" % (curr.name, play)
    elif act_num < 20:
        c = [1, 2, 3, 4, 5, "B","G","R","W","Y"]
        info = c[act_num - 10]
        opp.receive_clue(info)
        board.remove_clue()
        action_log = "Action %s gave clue: %s to %s" % (curr.name, str(info), opp.name)

    opp.see_players([curr])
    opp.see_board(board)
    curr.see_board(board)

    if deck.count() == 0:
        done = 1

    if board.finished==True:
        done = 1

    return (done, deck, curr, [opp], board, action_log)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
