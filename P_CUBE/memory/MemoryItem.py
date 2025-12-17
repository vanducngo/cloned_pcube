class MemoryItem:
    def __init__(self, sample, pseudo_label, uncertainty, feature=None):
        self.sample = sample
        self.feature = feature
        self.pseudo_label = pseudo_label
        self.uncertainty = uncertainty
        self.age = 0

    def increase_age(self, aging_speed):
        if not self.empty():
            self.age += aging_speed

    def get_data(self):
        return self.sample, self.uncertainty, self.age

    def empty(self):
        return self.sample == None
