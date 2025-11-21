class MemoryItem:
    def __init__(self, sample, pseudo_label, uncertainty, feature=None):
        self.sample = sample
        self.feature = feature
        self.pseudo_label = pseudo_label
        self.uncertainty = uncertainty
        self.age = 0

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"
