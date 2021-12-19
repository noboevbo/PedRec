import threading

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication

from pedrec.utils.input_providers.input_provider_base import InputProviderBase


class PedRecWorker(QThread):
    data_updated = pyqtSignal(int, np.ndarray, list, np.ndarray, int)
    input_provider: InputProviderBase = None

    def __init__(self, parent: QApplication):
        QThread.__init__(self, parent)
        self.window = parent
        self._lock = threading.Lock()
        self.active = True
        self.paused = False

    def _do_before_done(self):
        for i in range(2, 0, -1):
            self.msleep(100)
        print('Worker shut down.')

    def stop(self):
        print("Wait for worker to shut down...")
        self.active = False
        with self._lock:
            print("Acquired shut down lock")
            self._do_before_done()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def run(self):
        if self.input_provider is None:
            raise ValueError("No input provider set")
        frame_nr = 0
        iterator = self.input_provider.get_data()
        while True:
            with self._lock:
                if not self.active:
                    return
                if self.paused:
                    QThread.msleep(100)
                    continue
            img = next(iterator, None)
            if img is not None:
                frame_nr += 1
                self.run_impl(frame_nr, img)
            else:
                return
