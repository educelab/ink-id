'''Load, view, edit, and save datasets.'''

import os
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QObject, Slot, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QTreeView, QAbstractItemView, QWidget, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QFileDialog, QMessageBox


class DatasetError(RuntimeError):
    '''Represents an error with a Dataset.'''


class DatasetEditor(QWidget):
    saved = Signal(Path)

    def __init__(self, path: Path, parent: Optional[QObject]):
        super().__init__(parent)
        self._path = path
        self._tainted = False

        try:
            with open(self._path) as f:
                lines = f.readlines()
        except OSError as os_err:
            raise DatasetError(os_err) from os_err

        self.list_view = QListWidget()
        self.list_view.addItems([x.strip() for x in lines])
        self.list_view.currentRowChanged.connect(self.update_buttons)

        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self.new_item)

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

        self.btn_save = QPushButton("Save")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.btn_new)
        v_layout.addWidget(self.btn_add)
        v_layout.addWidget(self.btn_del)
        v_layout.addWidget(self.btn_up)
        v_layout.addWidget(self.btn_down)
        v_layout.addWidget(self.btn_save)
        v_layout.addStretch()

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.list_view)
        h_layout.addLayout(v_layout)
        self.setLayout(h_layout)

    def _get_items(self):
        items = []
        for row in range(self.list_view.count()):
            items.append(self.list_view.item(row).text())
        return items

    def _add_items(self, new_items):
        # Add the new items, but only if they aren't already in the list
        items = self._get_items()
        items_added = 0
        for filename in new_items:
            if filename not in items:
                relative_path = os.path.relpath(
                    filename, start=self._path.parents[0])
                self.list_view.addItem(str(relative_path))
                items_added += 1
        # Check if any items were actually added
        if items_added > 0:
            self._tainted = True
            self.btn_save.setEnabled(True)

    @Slot(int)
    def update_buttons(self, current: int):
        self.btn_del.setEnabled(
            current >= 0 and self.list_view.item(current) != None)
        self.btn_up.setEnabled(
            current >= 0 and self.list_view.item(current - 1) != None)
        self.btn_down.setEnabled(
            current >= 0 and self.list_view.item(current + 1) != None)
        self.btn_save.setEnabled(self._tainted)

    @Slot(bool)
    def new_item(self, checked: bool = True):
        filename = QFileDialog.getSaveFileName(
            self, 'New Dataset', dir=str(self._path.parents[0]), filter='Datasets (*.txt)')[0]
        if len(filename) < 1:
            return
        path = Path(filename).with_suffix('.txt')
        try:
            # Create (or truncate) the file
            with open(path, 'w'):
                pass
            self._add_items([str(path)])
        except OSError as err:
            QMessageBox.critical(self, 'Failed to save file', str(err))

    @Slot(bool)
    def add_item(self, checked: bool = False):
        filenames = QFileDialog.getOpenFileNames(
            self, 'Find Datasets and Datasources', dir=str(self._path.parents[0]), filter='Datasets and Datasources (*.txt *.json)')[0]
        self._add_items(filenames)

    @Slot(bool)
    def delete_item(self, checked: bool = False):
        self._tainted = True
        self.list_view.takeItem(self.list_view.currentRow())
        self.update_buttons(self.list_view.currentRow())

    @Slot(bool)
    def move_item_up(self, checked: bool = False):
        self._tainted = True
        current_row = self.list_view.currentRow()
        current_item = self.list_view.takeItem(current_row)
        self.list_view.insertItem(current_row - 1, current_item)
        self.list_view.setCurrentRow(current_row - 1)

    @Slot(bool)
    def move_item_down(self, checked: bool = False):
        self._tainted = True
        current_row = self.list_view.currentRow()
        current_item = self.list_view.takeItem(current_row)
        self.list_view.insertItem(current_row + 1, current_item)
        self.list_view.setCurrentRow(current_row + 1)

    @Slot(bool)
    def save(self, checked: bool = False):
        items = self._get_items()
        try:
            with open(self._path, 'w') as f:
                f.writelines([x + '\n' for x in items])
        except OSError as err:
            QMessageBox.critical(self, 'Failed to save file', str(err))
        self._tainted = False
        self.btn_save.setEnabled(False)
        self.saved.emit(self._path)

    def tainted(self):
        return self._tainted


class DatasourceEditor(QWidget):
    def __init__(self, path: Path, parent: Optional[QObject]):
        super().__init__(parent)
        self._path = path
        self._tainted = False

    def tainted(self):
        return self._tainted


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
        self._path = Path(filename)
        self._seen = []
        self._load_dataset(self.invisibleRootItem(), self._path)
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
        if str(path) in self._seen:
            raise DatasetError(f'Recursion loop detected on {path}')
        self._seen.append(str(path))
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

    def path(self):
        return self._path
