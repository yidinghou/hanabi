class Card:
    def __init__(self, value, color):
        self.value = value
        self.color = color

    def __str__(self):
        return (str(self.value) + self.color)


class Deck:
    def __init__(self, cards):
        self.cards = cards
        self.empty = False

    def __str__(self):
        return (str([str(c) for c in self.cards]))

    def count(self):
        return len(self.cards)

    def shuffle(self):
        random.shuffle(cards)

    def deal(self, n):
        if self.count() >= n:
            dealt = self.cards[0:n]
            self.cards = self.cards[n:]
            return (dealt)
        else:
            print("Deck is fully dealt")
            self.empty = True
            return ([])


class actions():
    def __init__(self):
        self.pi = dict.fromkeys(["Discard", "Play", "Give Clue"])
        self.pi["Discard"] = dict.fromkeys([0, 1, 2, 3], 0)
        self.pi["Play"] = dict.fromkeys([0, 1, 2, 3], 0)
        self.pi["Give Clue"] = dict.fromkeys([1, 2, 3, "R", "B", "Y"], 0)
        self.pi_array = [0] * 14

    def to_array(self):
        self.pi_array = list(self.pi["Discard"].values()) + list(self.pi["Play"].values()) + \
                        list(self.pi["Give Clue"].values())


class clue:
    def __init__(self, cards):
        self.clues = [0] * len(cards)
