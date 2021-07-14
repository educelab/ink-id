'''Create and configure the main window.'''

from PySide6.QtCore import Slot, Qt, QModelIndex
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QMainWindow, QLabel, QFileDialog, QHBoxLayout, QWidget, QSplitter, QMessageBox
from .datasets import DatasetModel, DatasetTreeView, DatasetError


class MainWindow(QMainWindow):
    '''Create and configure the main window.'''

    def __init__(self):
        super().__init__()
        self._create_menus()
        self._create_central_widget()
        self.setWindowTitle('Region Set Studio')
        # self.resize(QDesktopWidget().availableGeometry(self).size() * 0.75)
        self.showMaximized()

    def _create_menus(self):
        menu_bar = self.menuBar()
        # Menu: File
        menu_file = menu_bar.addMenu('&File')
        # Action: Open Data Set
        act_open_ds = menu_file.addAction('Open Dataset')
        act_open_ds.setShortcut(QKeySequence.Open)
        act_open_ds.triggered.connect(self.action_open_dataset)
        # Action: Quit
        act_quit = menu_file.addAction('Quit')
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.action_quit)

    def _create_central_widget(self):
        self.splitter = QSplitter()
        self.splitter.setOrientation(Qt.Vertical)
        self.dataset_tree = DatasetTreeView()
        self.dataset_tree.activated.connect(self.open_editor)
        self.splitter.addWidget(self.dataset_tree)
        self.setCentralWidget(self.splitter)

    @Slot(bool)
    def action_open_dataset(self, checked=False):
        '''Open a dataset and update the GUI for browsing it.'''
        filename = QFileDialog.getOpenFileName(parent=self,
                                               caption='Open Dataset',
                                               filter='Datasets (*.txt)')[0]
        try:
            self.dataset_model = DatasetModel(filename, parent=self)
            self.dataset_tree.setModel(self.dataset_model)
        except DatasetError as err:
            QMessageBox.critical(self, 'Error loading dataset', str(err))
            self.dataset_model = None

    @Slot(bool)
    def action_quit(self, checked=False):
        '''Close the main window, causing the application to exit.'''
        self.close()

    @Slot(QModelIndex)
    def open_editor(self, index):
        item = self.dataset_model.itemFromIndex(index)
        editor = item.editor(self)
        if (self.splitter.count() > 1):
            self.splitter.replaceWidget(1, editor)
        else:
            self.splitter.addWidget(editor)
