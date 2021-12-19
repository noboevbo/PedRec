from typing import List
import numpy as np
import pyqtgraph as pg

from pedrec.models.constants.action_mappings import ACTION


class ActionsBarChartView(pg.PlotWidget):
    def __init__(self, parent=None, **kargs):
        super().__init__(parent, **kargs)
        self.action_list = None
        self.__bg = None
        self.__labels = []
        self.setYRange(0, 1)

    def initialize_actions_chart(self, action_list: List[ACTION]):
        self.action_list = action_list

    def bg(self, **opts):
        if self.__bg == None:
            self.__bg = pg.BarGraphItem(x=range(0, len(self.action_list)), height=1, width=0.6)
            self.addItem(self.__bg)
            self.set_labels()
        return self.__bg

    def set_actions(self, action_probabilities: np.ndarray):
        self.bg().setOpts(height=action_probabilities)

    def set_labels(self):
        for idx, action in enumerate(self.action_list):
            text = pg.TextItem(
                text=action.name,
                angle=90,
                anchor=(0, 0.5), color=(255, 255, 255))
            self.addItem(text)
            text.setPos(idx, 0)
            self.__labels.append(text)

    def clear_data(self):
        self.bg().setOpts(height=0)

