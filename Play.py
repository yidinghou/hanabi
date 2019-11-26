import random
import numpy as np
from memory import Memory
from Game import *
from Player import *
import pickle
import config
from Model import *

def selfplay(model, EPISODES, memory, cards):
    scores = []
    colors = ["R","B","Y",'G',"W"]
    numbers = [1,1,1,2,2,3,3,4,4,5]

    cards = [Card(value, color) for color in colors for value in numbers]
    deck = Deck(cards = cards)

    for e in range(EPISODES):
        print(e)
        deck = Deck(cards=cards)
        p1 = Player("P1", deck.copy(), model)
        p2 = Player("P2", deck.copy(), model)

        board = Board(deck.copy())
        deck.shuffle()
        p1.draw(deck.deal(5))
        p2.draw(deck.deal(5))

        players = [p1, p2]
        turn = 0
        curr = players[(turn) % 2]
        opp = players[(turn + 1) % 2]
        currGS = GS(board, deck, curr, [opp], ["Start"], turn)
        done = 0

        while done == 0:
            curr = players[(turn) % 2]
            opp = players[(turn + 1) % 2]
            turn += 1
            curr.see_players([opp])
            curr.see_board(board)
            opp.see_players([curr])
            opp.see_board(board)

            output = p1.act(currGS, tau=1)
            currGS, value, done = currGS.takeAction(output[0])
            memory.commit_stmemory(currGS, output[1])

            if done == 1:
                if memory != None:
                    for move in memory.stmemory:
                        move["value"] = value

                memory.commit_ltmemory()

        scores.append(value)
    return (memory, scores)


memory = Memory(config.MEMORY_SIZE)
colors = ["R", "B", "Y", 'G', "W"]
numbers = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5]

cards = [Card(value, color) for color in colors for value in numbers]
deck = Deck(cards=cards)
model = nn_model()

p1 = Player("P1", deck.copy(), model)

selfplay(model, 2, memory, cards)
p1.replay(memory.ltmemory)
p1.model.get_weights()