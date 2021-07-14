'''Load, view, edit, and save datasets.'''

from pathlib import Path
from typing import Optional
from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QTreeView, QAbstractItemView, QWidget, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QFileDialog


class DatasetError(RuntimeError):
    '''Represents an error with a Dataset.'''


class DatasetEditor(QWidget):
    def __init__(self, path: Path, parent: Optional[QObject]):
        super().__init__(parent)
        self.path = path

        try:
            with open(self.path) as f:
                lines = f.readlines()
        except OSError as os_err:
            raise DatasetError(os_err) from os_err

        self.list_view = QListWidget()
        self.list_view.addItems([x.strip() for x in lines])
        self.list_view.currentRowChanged.connect(self.update_buttons)

        self.btn_add = QPushButton("+")
        self.btn_add.clicked.connect(self.add_item)

        self.btn_del = QPushButton("-")
        self.btn_del.setEnabled(False)
        self.btn_del.clicked.connect(self.delete_item)

        self.btn_up = QPushButton("^")
        self.btn_up.setEnabled(False)
        self.btn_up.clicked.connect(self.move_item_up)

        self.btn_down = QPushButton("v")
        self.btn_down.setEnabled(False)
        self.btn_down.clicked.connect(self.move_item_down)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.btn_add)
        v_layout.addWidget(self.btn_del)
        v_layout.addWidget(self.btn_up)
        v_layout.addWidget(self.btn_down)
        v_layout.addStretch()

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.list_view)
        h_layout.addLayout(v_layout)
        self.setLayout(h_layout)

    @Slot(int)
    def update_buttons(self, current: int):
        self.btn_del.setEnabled(
            current >= 0 and self.list_view.item(current) != None)
        self.btn_up.setEnabled(
            current >= 0 and self.list_view.item(current - 1) != None)
        self.btn_down.setEnabled(
            current >= 0 and self.list_view.item(current + 1) != None)

    @Slot(bool)
    def add_item(self, checked: bool = False):
        # Get an existing list of items in the list
        items = []
        for row in range(self.list_view.count()):
            items.append(self.list_view.item(row).text())
        filenames = QFileDialog.getOpenFileNames(
            self, 'Find Datasets and Datasources', filter='Datasets and Datasources (*.txt *.json)')[0]
        # Add the newly selected items, but only if they aren't already in the list
        for filename in filenames:
            if filename not in items:
                self.list_view.addItem(filename)

    @Slot(bool)
    def delete_item(self, checked: bool = False):
        self.list_view.takeItem(self.list_view.currentRow())
        self.update_buttons(self.list_view.currentRow())

    @Slot(bool)
    def move_item_up(self, checked: bool = False):
        current_row = self.list_view.currentRow()
        current_item = self.list_view.takeItem(current_row)
        self.list_view.insertItem(current_row - 1, current_item)
        self.list_view.setCurrentRow(current_row - 1)

    @Slot(bool)
    def move_item_down(self, checked: bool = False):
        current_row = self.list_view.currentRow()
        current_item = self.list_view.takeItem(current_row)
        self.list_view.insertItem(current_row + 1, current_item)
        self.list_view.setCurrentRow(current_row + 1)


class DatasourceEditor(QWidget):
    def __init__(self, path: Path, parent: Optional[QObject]):
        super().__init__(parent)
        self.path = path


class DatasetTreeView(QTreeView):
    def __init__(self):
        super().__init__()
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setHeaderHidden(True)


class DatasetItem(QStandardItem):
    def __init__(self, path: Path):
        super().__init__(str(path))
        self.path = path

    def type(self):
        return QStandardItem.UserType + 1

    def editor(self, parent: QObject):
        return DatasetEditor(self.path, parent=parent)


class DatasourceItem(QStandardItem):
    def __init__(self, path: Path):
        super().__init__(str(path))
        self.path = path

    def type(self):
        return QStandardItem.UserType + 2

    def editor(self, parent: QObject):
        return DatasourceEditor(self.path, parent=parent)


class DatasetModel(QStandardItemModel):
    def __init__(self, filename: str, parent: Optional[QObject]):
        super().__init__(parent)
        self.path = Path(filename)
        self.seen = []
        self._load_dataset(self.invisibleRootItem(), self.path)
        self.setHorizontalHeaderLabels(['Path'])

    def _load_dataset(self, parent_item: QStandardItem, path: Path, relative_to: Path = None):
        '''Given the path to a dataset, recursively build a dataset/datasource tree.'''
        # Normalize and resolve the path before processing
        if path.is_absolute():
            path = path.resolve()
        elif relative_to:
            path = relative_to.joinpath(path).resolve()
        else:
            raise DatasetError('Relative path given without context')
        # JSON files are datasources and should end the recursion
        if path.suffix == '.json':
            datasource_item = DatasourceItem(path)
            parent_item.appendRow(datasource_item)
            return
        # Make sure this item hasn't been seen before, and mark it as seen now
        if str(path) in self.seen:
            raise DatasetError(f'Recursion loop detected on {path}')
        self.seen.append(str(path))
        # All other files are to be treated as datasets with one or more children
        dataset_item = DatasetItem(path)
        parent_item.appendRow(dataset_item)
        try:
            with open(path) as f:
                for line in f.readlines():
                    self._load_dataset(dataset_item, Path(
                        line.strip()), relative_to=path.parents[0])
        except OSError as os_err:
            raise DatasetError(os_err) from os_err
