'''Create and configure the main window.'''

from pathlib import Path
from PySide6.QtCore import Slot, Qt, QModelIndex
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QMainWindow, QFileDialog, QSplitter, QMessageBox
from .datasets import DatasetModel, DatasetTreeView, DatasetError, DatasetEditor


class MainWindow(QMainWindow):
    '''Create and configure the main window.'''

    def __init__(self):
        super().__init__()
        self.dataset_model = None
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
        self.act_new_ds = menu_file.addAction('New Dataset')
        self.act_new_ds.setShortcut(QKeySequence.New)
        self.act_new_ds.triggered.connect(self.action_new_dataset)
        # Action: Open Data Set
        self.act_open_ds = menu_file.addAction('Open Dataset')
        self.act_open_ds.setShortcut(QKeySequence.Open)
        self.act_open_ds.triggered.connect(self.action_open_dataset)
        # Action: Close Data Set
        self.act_close_ds = menu_file.addAction('Close Dataset')
        self.act_close_ds.setShortcut(QKeySequence.Close)
        self.act_close_ds.triggered.connect(self.action_close_dataset)
        self.act_close_ds.setEnabled(False)
        # Action: Quit
        self.act_quit = menu_file.addAction('Quit')
        self.act_quit.setShortcut(QKeySequence.Quit)
        self.act_quit.triggered.connect(self.action_quit)

    def _create_central_widget(self):
        self.splitter = QSplitter()
        self.splitter.setOrientation(Qt.Vertical)
        self.dataset_tree = DatasetTreeView()
        self.dataset_tree.activated.connect(self.open_editor)
        self.splitter.addWidget(self.dataset_tree)
        self.setCentralWidget(self.splitter)

    def _load_dataset(self, filename: str):
        try:
            self.dataset_model = DatasetModel(filename, parent=self)
            self.dataset_tree.setModel(self.dataset_model)
            self.dataset_tree.expandAll()
            self.act_new_ds.setEnabled(False)
            self.act_open_ds.setEnabled(False)
            self.act_close_ds.setEnabled(True)
        except DatasetError as err:
            QMessageBox.critical(self, 'Error loading dataset', str(err))
            self.dataset_model = None

    def _safe_to_close(self):
        if self.splitter.count() > 1:
            editor = self.splitter.widget(1)
            if editor.tainted():
                discard_yorn = QMessageBox.question(self, 'Discard unsaved changes?',
                                                    'You have unsaved changes. Are you sure you want to discard them?')
                if discard_yorn == QMessageBox.No:
                    return False
        return True

    @Slot(bool)
    def action_new_dataset(self, checked: bool = True):
        filename = QFileDialog.getSaveFileName(
            self, 'New Dataset', filter='Datasets (*.txt)')[0]
        if len(filename) < 1:
            return
        path = Path(filename).with_suffix('.txt')
        try:
            # Create (or truncate) the file
            with open(path, 'w'):
                pass
            # Then load up that dataset file
            self._load_dataset(str(path))
        except OSError as err:
            QMessageBox.critical(self, 'Failed to save file', str(err))

    @Slot(bool)
    def action_open_dataset(self, checked=False):
        '''Open a dataset and update the GUI for browsing it.'''
        filename = QFileDialog.getOpenFileName(parent=self,
                                               caption='Open Dataset',
                                               filter='Datasets (*.txt)')[0]
        if len(filename) < 1:
            return
        self._load_dataset(filename)

    @Slot(bool)
    def action_close_dataset(self, checked=False):
        '''Close a currently open dataset.'''
        if not self._safe_to_close():
            return
        # We need to destroy the second widget from the splitter
        if self.splitter.count() > 1:
            editor = self.splitter.widget(1)
            editor.setParent(None)
        self.dataset_tree.setModel(None)
        self.dataset_model = None
        self.act_new_ds.setEnabled(True)
        self.act_open_ds.setEnabled(True)
        self.act_close_ds.setEnabled(False)

    @Slot(bool)
    def action_quit(self, checked=False):
        '''Close the main window, causing the application to exit.'''
        if self._safe_to_close():
            self.close()

    @Slot(Path)
    def reload_dataset(self, path: Path):
        filename = str(self.dataset_model.path())
        self.action_close_dataset()
        self._load_dataset(filename)

    @Slot(QModelIndex)
    def open_editor(self, index):
        if not self._safe_to_close():
            return
        item = self.dataset_model.itemFromIndex(index)
        editor = item.editor(self)
        if isinstance(editor, DatasetEditor):
            editor.saved.connect(self.reload_dataset)
        if (self.splitter.count() > 1):
            self.splitter.replaceWidget(1, editor)
        else:
            self.splitter.addWidget(editor)
