class MemoryItem:
    def __init__(self, sample, pseudo_label, uncertainty, odp_score = 0):
        self.sample = sample
        self.pseudo_label = pseudo_label
        self.uncertainty = uncertainty
        self.odp_score = odp_score # Lưu ODP Score của mẫu
        self.age = 0

    def increase_age(self, aging_speed):
        if not self.empty():
            self.age += aging_speed

    def get_data(self):
        return self.sample, self.uncertainty, self.age, self.odp_score

    def empty(self):
        return self.sample == None
