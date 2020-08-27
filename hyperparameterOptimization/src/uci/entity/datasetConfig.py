from src.uci.util.task_enum import TaskType


class DatasetConfig:

    def __init__(self, name, folder, file_name, task, prediction_column):
        self.name: str = name
        self.folder: str = folder
        self.file_name: str = file_name
        self.task: TaskType = TaskType(int(task))
        self.prediction_column: int = prediction_column
        self.rows: int = 0
        self.features: int = 0
        self.n_in = 0
        self.n_out = 0

