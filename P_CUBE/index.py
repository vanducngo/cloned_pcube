from .memory.MemoryItem import MemoryItem
from .filter.index import P_Cube_Filter
from .memory.PCubeMemoryBank import PCubeMemoryBank

class P_CUBE:
    # def __init__(self, original_model, student_model, teacher_model):
    #     self.filter = P_Cube_Filter(self)
    #     self.memory = PCubeMemoryBank(self)

    #     self.original_model = original_model
    #     self.student_model = student_model
    #     self.teacher_model = teacher_model

    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.filter = P_Cube_Filter(source_model=None) #TODO: Source model
        self.memory = PCubeMemoryBank(capacity, num_class, lambda_t, lambda_u)
        self.student_model = None

    def add_instance(self, memory_item: MemoryItem):
        isCleanSample = self.filter.isSampleClean(memory_item, self.student_model)

        # Sample are not valid to be added to the memory bank
        if not isCleanSample:
            return
        
        self.memory.add_instance(memory_item)

    def get_memory(self):
        return self.memory.get_memory()
    
    def get_occupancy(self):
        return self.memory.get_occupancy()