'''Load, view, edit, and save datasets.'''

import os
import json
from json.decoder import JSONDecodeError
import re
from pathlib import Path
from typing import Optional, List
from PySide6.QtCore import QObject, Slot, Signal, Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem, QPixmap, QPen, QBrush
from PySide6.QtWidgets import QTreeView, QAbstractItemView, QWidget, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QGraphicsView, QSplitter, QFormLayout, QComboBox, QLabel, QCheckBox, QDialog, QGraphicsScene, QListWidgetItem, QGraphicsRectItem, QFrame, QGroupBox, QSpinBox, QGraphicsItem


def parse_ppm_header(filename):
    comments_re = re.compile('^#')
    width_re = re.compile('^width')
    height_re = re.compile('^height')
    dim_re = re.compile('^dim')
    ordered_re = re.compile('^ordered')
    type_re = re.compile('^type')
    version_re = re.compile('^version')
    header_terminator_re = re.compile('^<>$')

    with open(filename, 'rb') as f:
        while True:
            line = f.readline().decode('utf-8')
            if comments_re.match(line):
                pass
            elif width_re.match(line):
                width = int(line.split(': ')[1])
            elif height_re.match(line):
                height = int(line.split(': ')[1])
            elif dim_re.match(line):
                dim = int(line.split(': ')[1])
            elif ordered_re.match(line):
                ordered = line.split(': ')[1].strip() == 'true'
            elif type_re.match(line):
                val_type = line.split(': ')[1].strip()
                assert val_type in ['double']
            elif version_re.match(line):
                version = line.split(': ')[1].strip()
            elif header_terminator_re.match(line):
                break

    return {
        'width': width,
        'height': height,
        'dim': dim,
        'ordered': ordered,
        'type': val_type,
        'version': version
    }


class DatasetError(RuntimeError):
    '''Represents an error with a Dataset.'''


class Datasource:
    def __init__(self, path: Path, schema_version='0.1', type_='volume'):
        self._path = path
        self.setSchemaVersion(schema_version)
        self.setType(type_)
        self.setVolume(None)
        self.setPPM(None)
        self.setMask(None)
        self.setInkLabel(None)
        self.setRGBLabel(None)
        self.setVCTLabel(None)
        self.setInvertNormals(False)
        self.setBoundingBox(None)

    @staticmethod
    def fromPath(path: Path):
        datasource = Datasource(path)
        try:
            with open(path) as f:
                data = json.load(f)
                datasource.setSchemaVersion(data['schema_version'])
                datasource.setType(data['type'])
                datasource.setVolume(data['volume'])
                if datasource.getType() == 'region':
                    datasource.setPPM(data['ppm'])
                    datasource.setMask(data['mask'])
                    datasource.setInkLabel(data['ink_label'])
                    datasource.setRGBLabel(data['rgb_label'])
                    datasource.setVCTLabel(data['volcart_texture_label'])
                    datasource.setInvertNormals(data['invert_normals'])
                    datasource.setBoundingBox(data['bounding_box'])
                return datasource
        except OSError as os_error:
            raise DatasetError(
                f'Failed to open datasource: {path}') from os_error
        except JSONDecodeError as json_error:
            raise DatasetError(
                f'Failed to parse JSON for datasource: {path}') from json_error
        except KeyError as key_error:
            raise DatasetError(
                f'Expected element is missing: {key_error}') from key_error

    def _absolute_path(self, path: str) -> str:
        if path is None:
            return None
        real_path = Path(path)
        if real_path.is_absolute():
            return str(real_path)
        return str(self._path.parents[0].joinpath(real_path).resolve())

    def save(self):
        data = {}
        data['schema_version'] = self.getSchemaVersion()
        data['type'] = self.getType()
        data['volume'] = self.getVolume()
        if self.getType() == 'region':
            data['ppm'] = self.getPPM()
            data['invert_normals'] = self.getInvertNormals()
            data['mask'] = self.getMask()
            data['ink_label'] = self.getInkLabel()
            data['rgb_label'] = self.getRGBLabel()
            data['volcart_texture_label'] = self.getVCTLabel()
            data['bounding_box'] = self.getBoundingBox()
        data_out = json.dumps(data, indent=4)
        try:
            with open(self._path, 'w') as f:
                f.write(data_out)
                f.write('\n')
        except OSError as err:
            QMessageBox.critical(self, 'Failed to save file', str(err))

    def makeRelative(self, path: str) -> str:
        return os.path.relpath(path, start=self._path.parents[0])

    def getPath(self) -> Path:
        return self._path

    def setSchemaVersion(self, value: str):
        if value is None or value not in '0.1':
            raise DatasetError(f'Schema not supported: {value}')
        self._schema_version = value

    def getSchemaVersion(self) -> str:
        return self._schema_version

    def setType(self, type_: str):
        if type_ is None or type_ not in ('region', 'volume'):
            raise DatasetError(f'Invalid type: {type_}')
        self._type = type_

    def getType(self) -> str:
        return self._type

    def setVolume(self, value: str):
        if value is None or len(value) < 1:
            self._volume = None
        else:
            self._volume = value

    def getVolume(self, absolute: bool = False) -> str:
        return self._absolute_path(self._volume) if absolute else self._volume

    def setPPM(self, ppm: str):
        if ppm is None or len(ppm) < 1:
            self._ppm = None
        else:
            self._ppm = ppm

    def getPPM(self, absolute: bool = False) -> str:
        return self._absolute_path(self._ppm) if absolute else self._ppm

    def setMask(self, mask: str):
        if mask is None or len(mask) < 1:
            self._mask = None
        else:
            self._mask = mask

    def getMask(self, absolute: bool = False) -> str:
        return self._absolute_path(self._mask) if absolute else self._mask

    def setInkLabel(self, ink_label: str):
        if ink_label is None or len(ink_label) < 1:
            self._ink_label = None
        else:
            self._ink_label = ink_label

    def getInkLabel(self, absolute: bool = False) -> str:
        return self._absolute_path(self._ink_label) if absolute else self._ink_label

    def setRGBLabel(self, rgb_label: str):
        if rgb_label is None or len(rgb_label) < 1:
            self._rgb_label = None
        else:
            self._rgb_label = rgb_label

    def getRGBLabel(self, absolute: bool = False) -> str:
        return self._absolute_path(self._rgb_label) if absolute else self._rgb_label

    def setVCTLabel(self, vct_label: str):
        if vct_label is None or len(vct_label) < 1:
            self._vct_label = None
        else:
            self._vct_label = vct_label

    def getVCTLabel(self, absolute: bool = False) -> str:
        return self._absolute_path(self._vct_label) if absolute else self._vct_label

    def setInvertNormals(self, invert_normals: bool):
        if invert_normals is None:
            self._invert_normals = False
        else:
            self._invert_normals = invert_normals

    def getInvertNormals(self) -> bool:
        return self._invert_normals

    def setBoundingBox(self, bounding_box: List[int]):
        if bounding_box is None or len(bounding_box) < 1:
            self._bounding_box = None
        else:
            if len(bounding_box) != 4:
                raise DatasetError(f'Invalid bounding box: {bounding_box}')
            self._bounding_box = bounding_box

    def getBoundingBox(self) -> List[int]:
        return self._bounding_box


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
        filename, filter_ = QFileDialog.getSaveFileName(
            self, 'New Dataset or Datasource', dir=str(self._path.parents[0]), filter='Datasets (*.txt);;Datasources (*.json)')
        if len(filename) < 1:
            return
        try:
            if filter_.startswith('Datasets'):
                path = Path(filename).with_suffix('.txt')
                # Create (or truncate) the file
                with open(path, 'w'):
                    pass
            else:
                path = Path(filename).with_suffix('.json')
                # Create (or truncate) the file with a new generic Datasource
                datasource = Datasource(path)
                datasource.save()
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


class FileBrowserWidget(QWidget):
    changed = Signal(str)

    def __init__(self, value: str, dir_: str, filter_: str):
        # filter_ = None implies directory
        super().__init__()
        self.value = value
        self.dir = dir_
        self.filter = filter_
        self.label = QLabel(value if value else 'None')
        self.browse_btn = QPushButton('Browse')
        self.browse_btn.clicked.connect(self.browse_file)
        self.remove_btn = QPushButton(text='X', parent=self)
        self.remove_btn.clicked.connect(self.remove_file)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label, 1)
        h_layout.addWidget(self.browse_btn)
        h_layout.addWidget(self.remove_btn)
        h_layout.setContentsMargins(0, 0, 0, 0)
        self.remove_btn.setVisible(self.value is not None)
        self.setLayout(h_layout)

    @Slot(bool)
    def browse_file(self, checked: bool = False):
        if self.filter is None:
            filename = QFileDialog.getExistingDirectory(
                self, 'Browse Directory', self.dir)
        else:
            filename = QFileDialog.getOpenFileName(
                self, 'Browse File', self.dir, self.filter)[0]
        if len(filename) < 1:
            return
        relative_filename = os.path.relpath(filename, start=self.dir)
        self.value = relative_filename
        self.label.setText(self.value)
        self.remove_btn.setVisible(True)
        self.changed.emit(self.value)

    @Slot(bool)
    def remove_file(self, checked: bool = False):
        self.value = None
        self.label.setText('None')
        self.remove_btn.setVisible(False)
        self.changed.emit(self.value)


class BoundingBoxGraphicsRect(QGraphicsRectItem):
    def __init__(self):
        super().__init__()
        # self.setAcceptHoverEvents(True)
        # self.setFlag(QGraphicsItem.ItemIsMovable, True)

    def change_pos(self, x1, y1, x2, y2):
        self.prepareGeometryChange()
        self.setRect(x1, y1, x2 - x1, y2 - y1)


class RegionBoundsDialog(QDialog):
    def __init__(self, parent, datasource: Datasource, ghosts: list):
        super().__init__(parent, Qt.Dialog)
        self._datasource = datasource
        self._ghosts = ghosts
        self._ppm_data = parse_ppm_header(self._datasource.getPPM(True))
        self._tainted = False
        self._have_saved = False
        self._selected_ghost_path = None
        self._ghost_edit_on_close = False

        self.setWindowTitle('Edit Bounding Box')

        self.btn_apply = QPushButton('Apply')
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.action_apply)
        btn_close = QPushButton('Close')
        btn_close.clicked.connect(self.action_close)
        self.btn_close_edit_ghost = QPushButton('Close && Edit Ghost')
        self.btn_close_edit_ghost.setEnabled(False)
        self.btn_close_edit_ghost.clicked.connect(self.action_close_edit_ghost)
        btns_layout = QHBoxLayout()
        btns_layout.addWidget(self.btn_apply)
        btns_layout.addWidget(btn_close)
        btns_layout.addWidget(self.btn_close_edit_ghost)

        pixmap = QPixmap()
        scene = QGraphicsScene()
        scene.setSceneRect(
            0, 0, self._ppm_data['width'], self._ppm_data['height'])
        scene.setBackgroundBrush(QBrush(Qt.gray))
        scene.addRect(scene.sceneRect(), QPen(Qt.NoPen), QBrush(Qt.white))
        self.pixmap_item = scene.addPixmap(pixmap)

        bbox_pen = QPen()
        bbox_pen.setColor(Qt.green)
        self.bounding_box_gfx = BoundingBoxGraphicsRect()
        self.bounding_box_gfx.setPen(bbox_pen)

        group_bounds = QGroupBox('Bounding Box')
        bounds_layout = QFormLayout()
        self.bounds_x1 = QSpinBox()
        self.bounds_x1.setMaximum(self._ppm_data['width'] - 2)
        self.bounds_x1.valueChanged.connect(self.change_pos)
        self.bounds_y1 = QSpinBox()
        self.bounds_y1.setMaximum(self._ppm_data['height'] - 2)
        self.bounds_y1.valueChanged.connect(self.change_pos)
        self.bounds_x2 = QSpinBox()
        self.bounds_x2.setMaximum(self._ppm_data['width'] - 1)
        self.bounds_x2.valueChanged.connect(self.change_pos)
        self.bounds_y2 = QSpinBox()
        self.bounds_y2.setMaximum(self._ppm_data['height'] - 1)
        self.bounds_y2.valueChanged.connect(self.change_pos)
        existing_bounds = self._datasource.getBoundingBox()
        if existing_bounds is None:
            self.bounds_x1.setValue(0)
            self.bounds_y1.setValue(0)
            self.bounds_x2.setValue(self._ppm_data['width'] - 1)
            self.bounds_x2.setMinimum(1)
            self.bounds_y2.setValue(self._ppm_data['height'] - 1)
            self.bounds_y2.setMinimum(1)
            self._saved_value = [
                0, 0, self._ppm_data['width'] - 1, self._ppm_data['height'] - 1]
        else:
            self._saved_value = existing_bounds
            bx, by, bx2, by2 = existing_bounds
            self.bounds_x1.setValue(bx)
            self.bounds_y1.setValue(by)
            self.bounds_x2.setValue(bx2)
            self.bounds_x2.setMinimum(bx + 1)
            self.bounds_y2.setValue(by2)
            self.bounds_y2.setMinimum(by + 1)
            # Reset taint in this case when setting existing bounds so it
            # doesn't trigger the "Apply" button when no real change has been
            # made. When no bounds exist (the other section of this if
            # condition), we leave the save button for the auto-generated
            # bounds.
            self._tainted = False
            self.btn_apply.setEnabled(False)
        bounds_layout.addRow('Left', self.bounds_x1)
        bounds_layout.addRow('Top', self.bounds_y1)
        bounds_layout.addRow('Right', self.bounds_x2)
        bounds_layout.addRow('Bottom', self.bounds_y2)
        group_bounds.setLayout(bounds_layout)

        self.underlay_list = QListWidget()
        self.underlay_list.currentItemChanged.connect(
            self.underlay_selection_changed)
        for underlay in [self._datasource.getMask(True), self._datasource.getInkLabel(True), self._datasource.getRGBLabel(True)]:
            if underlay is None:
                continue
            relpath = self._datasource.makeRelative(underlay)
            item = QListWidgetItem(relpath)
            pixmap = QPixmap(underlay)
            item.setData(Qt.UserRole + 1, pixmap)
            self.underlay_list.addItem(item)

        self.browse_viz = QPushButton('Browse underlay images...')
        self.browse_viz.clicked.connect(self.browse_underlays)

        group_underlays = QGroupBox('Visualization Underlays')
        underlay_layout = QVBoxLayout()
        underlay_layout.addWidget(self.browse_viz)
        underlay_layout.addWidget(self.underlay_list)
        group_underlays.setLayout(underlay_layout)

        ghost_list = QListWidget()
        ghost_list.itemChanged.connect(self.ghost_changed)
        ghost_list.currentItemChanged.connect(self.ghost_selection_changed)
        self.ghost_pen = QPen()
        self.ghost_pen.setColor(Qt.red)
        self.ghost_pen_selected = QPen()
        self.ghost_pen_selected.setColor(Qt.blue)
        for ghost_path in self._ghosts:
            try:
                ds = Datasource.fromPath(ghost_path)
                relpath = self._datasource.makeRelative(ghost_path)
                item = QListWidgetItem(f'{ds.getBoundingBox()} from {relpath}')
                item.setCheckState(Qt.Checked)
                ghost_list.addItem(item)

                bx, by, bx2, by2 = ds.getBoundingBox()

                rect = QGraphicsRectItem(bx, by, bx2 - bx, by2 - by)
                rect.setPen(self.ghost_pen)
                scene.addItem(rect)
                item.setData(Qt.UserRole + 1, rect)
                item.setData(Qt.UserRole + 2, ghost_path)
            except DatasetError:
                pass

        scene.addItem(self.bounding_box_gfx)

        group_ghosts = QGroupBox('Ghosts')
        ghost_layout = QVBoxLayout()
        ghost_layout.addWidget(ghost_list)
        group_ghosts.setLayout(ghost_layout)

        v_layout = QVBoxLayout()
        v_layout.addWidget(group_bounds)
        v_layout.addWidget(group_underlays)
        v_layout.addWidget(group_ghosts)
        v_layout.addLayout(btns_layout)

        generic = QWidget()
        generic.setLayout(v_layout)

        gfx = QGraphicsView()
        gfx.setScene(scene)

        splitter = QSplitter()
        splitter.addWidget(gfx)
        splitter.addWidget(generic)

        h_layout = QHBoxLayout()
        h_layout.addWidget(splitter)

        self.setLayout(h_layout)

    @Slot(QListWidgetItem)
    def ghost_changed(self, item: QListWidgetItem):
        checked = item.checkState() == Qt.Checked
        data = item.data(Qt.UserRole + 1)
        data.setVisible(checked)

    @Slot(QListWidgetItem, QListWidgetItem)
    def ghost_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        if previous is not None:
            data = previous.data(Qt.UserRole + 1)
            data.setPen(self.ghost_pen)
        if current is not None:
            data = current.data(Qt.UserRole + 1)
            data.setPen(self.ghost_pen_selected)
            full_path = current.data(Qt.UserRole + 2)
            self._selected_ghost_path = current.data(Qt.UserRole + 2)
            self.btn_close_edit_ghost.setEnabled(True)
        else:
            self._selected_ghost_path = None
            self.btn_close_edit_ghost.setEnabled(False)

    @Slot(QListWidgetItem, QListWidgetItem)
    def underlay_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        data = current.data(Qt.UserRole + 1)
        self.pixmap_item.setPixmap(data)

    @Slot(bool)
    def action_apply(self, checked: bool = False):
        self._have_saved = True
        self._saved_value = [self.bounds_x1.value(), self.bounds_y1.value(
        ), self.bounds_x2.value(), self.bounds_y2.value()]
        self._tainted = False
        self.btn_apply.setEnabled(False)

    def _safe_to_close(self) -> bool:
        if self._tainted:
            discard_yorn = QMessageBox.question(self, 'Discard unsaved changes?',
                                                'You have unsaved changes. Are you sure you want to discard them?')
            if discard_yorn == QMessageBox.No:
                return False
        return True

    def _close_dialog(self):
        # Change return code depending if we have saved or not
        if self._have_saved:
            self.accept()
        else:
            self.reject()

    @Slot(bool)
    def action_close(self, checked: bool = False):
        if not self._safe_to_close():
            return
        self._close_dialog()

    @Slot(bool)
    def action_close_edit_ghost(self, checked: bool = False):
        if not self._safe_to_close():
            return
        self._ghost_edit_on_close = True
        self._close_dialog()

    def _update_rect_size_constraints(self):
        self.bounds_x1.setMinimum(0)
        self.bounds_x1.setMaximum(self._ppm_data['width'] - 2)

        self.bounds_y1.setMinimum(0)
        self.bounds_y1.setMaximum(self._ppm_data['height'] - 2)

        self.bounds_x2.setMinimum(self.bounds_x1.value() + 1)
        self.bounds_x2.setMaximum(self._ppm_data['width'] - 1)

        self.bounds_y2.setMinimum(self.bounds_y1.value() + 1)
        self.bounds_y2.setMaximum(self._ppm_data['height'] - 1)

    @Slot(bool)
    def browse_underlays(self, checked: bool = False):
        filenames = QFileDialog.getOpenFileNames(self, 'Browse Images', str(self._datasource.getPath(
        ).parents[0]), 'Image Files (*.png *.tif *.tiff *.jpg *.jpeg)')[0]
        cnt = self.underlay_list.count()
        for filename in filenames:
            relpath = self._datasource.makeRelative(filename)
            found_dup = False
            for i in range(cnt):
                x = self.underlay_list.item(i)
                if x.text() == relpath:
                    found_dup = True
            if found_dup:
                continue
            item = QListWidgetItem(relpath)
            pixmap = QPixmap(filename).copy(
                0, 0, self._ppm_data['width'], self._ppm_data['height'])
            item.setData(Qt.UserRole + 1, pixmap)
            self.underlay_list.addItem(item)

    @Slot(int)
    def change_pos(self, value):
        self._tainted = True
        self.btn_apply.setEnabled(True)
        self._update_rect_size_constraints()
        self.bounding_box_gfx.change_pos(self.bounds_x1.value(
        ), self.bounds_y1.value(), self.bounds_x2.value(), self.bounds_y2.value())

    def value(self):
        return self._saved_value

    def ghost_to_edit(self):
        return self._selected_ghost_path if self._ghost_edit_on_close else None


class RegionBoundsWidget(QWidget):
    changed = Signal(list)
    edit_ghost = Signal(str)

    def __init__(self, datasource: Datasource, datasources_paths: list):
        super().__init__()
        self.datasource = datasource
        self.datasources_paths = datasources_paths
        self.value = datasource.getBoundingBox()
        self._update_ghosts()

        self.label = QLabel(str(self.value))
        self.edit_btn = QPushButton('Edit')
        self.edit_btn.clicked.connect(self.edit_bounds)
        self.remove_btn = QPushButton(text='X', parent=self)
        self.remove_btn.clicked.connect(self.remove_bounds)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.label, 1)
        h_layout.addWidget(self.edit_btn)
        h_layout.addWidget(self.remove_btn)
        h_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_btn.setEnabled(self.datasource.getPPM() is not None)
        self.remove_btn.setVisible(self.value is not None)
        self.setLayout(h_layout)

    def launch_bounds_editor(self):
        if self.edit_btn.isEnabled():
            self.edit_bounds()

    def _update_ghosts(self):
        ghosts = []
        ppm = self.datasource.getPPM(True)
        if ppm is None:
            self._ghosts = ghosts
            return
        for ds_path in self.datasources_paths:
            # Invalid datasources are skipped
            try:
                ds = Datasource.fromPath(Path(ds_path))
            except DatasetError:
                continue
            # We skip ourself also
            if ds.getPath() == self.datasource.getPath():
                continue
            # We add datasets which have a PPM matching us
            if ppm == ds.getPPM(True):
                ghosts.append(ds_path)
        self._ghosts = ghosts

    def ppm_changed(self):
        self.edit_btn.setEnabled(self.datasource.getPPM() is not None)
        self._update_ghosts()

    @Slot(bool)
    def edit_bounds(self, checked: bool = False):
        dialog = RegionBoundsDialog(self, self.datasource, self._ghosts)
        if dialog.exec() == QDialog.Accepted:
            self.value = dialog.value()
            self.label.setText(str(self.value))
            self.remove_btn.setVisible(True)
            self.changed.emit(self.value)
        ghost_to_edit = dialog.ghost_to_edit()
        if ghost_to_edit is not None:
            self.edit_ghost.emit(str(ghost_to_edit))

    @Slot(bool)
    def remove_bounds(self, checked: bool = False):
        self.value = None
        self.label.setText(str(self.value))
        self.remove_btn.setVisible(False)
        self.changed.emit(self.value)


class DatasourceEditor(QWidget):
    switch_editors = Signal(str)

    def __init__(self, path: Path, datasources_paths: list, parent: Optional[QObject]):
        super().__init__(parent)
        self._path = path
        self._tainted = False
        self._datasource = Datasource.fromPath(path)
        self._datasources_paths = datasources_paths

        self.ds_schema_version = QLabel(self._datasource.getSchemaVersion())

        self.ds_type = QComboBox()
        self.ds_type.addItem('volume')
        self.ds_type.addItem('region')
        self.ds_type.setCurrentIndex(
            1 if self._datasource.getType() == 'region' else 0)
        self.ds_type.currentTextChanged.connect(self.update_type)

        self.ds_volume = FileBrowserWidget(self._datasource.getVolume(), str(
            self._datasource.getPath().parents[0]), None)
        self.ds_volume.changed.connect(self.update_volume)
        self.ds_ppm = FileBrowserWidget(self._datasource.getPPM(), str(
            self._datasource.getPath().parents[0]), 'PPM Files (*.ppm)')
        self.ds_ppm.changed.connect(self.update_ppm)
        self.ds_mask = FileBrowserWidget(self._datasource.getMask(), str(
            self._datasource.getPath().parents[0]), 'Image Files (*.png *.tiff *.tif)')
        self.ds_mask.changed.connect(self.update_mask)
        self.ds_ink_label = FileBrowserWidget(self._datasource.getInkLabel(), str(
            self._datasource.getPath().parents[0]), 'Image Files (*.png *.tiff *.tif)')
        self.ds_ink_label.changed.connect(self.update_ink_label)
        self.ds_rgb_label = FileBrowserWidget(self._datasource.getRGBLabel(), str(
            self._datasource.getPath().parents[0]), 'Image Files (*.png *.tiff *.tif)')
        self.ds_rgb_label.changed.connect(self.update_rgb_label)
        self.ds_vct_label = FileBrowserWidget(self._datasource.getVCTLabel(), str(
            self._datasource.getPath().parents[0]), 'Image Files (*.png *.tiff *.tif)')
        self.ds_vct_label.changed.connect(self.update_vct_label)

        self.ds_invert_normals = QCheckBox()
        self.ds_invert_normals.setChecked(self._datasource.getInvertNormals())
        self.ds_invert_normals.stateChanged.connect(self.update_invert_normals)

        self.ds_bounding_box = RegionBoundsWidget(
            self._datasource, self._datasources_paths)
        self.ds_bounding_box.changed.connect(self.update_bounding_box)
        self.ds_bounding_box.edit_ghost.connect(self.switch_files)

        self.form_layout = QFormLayout()
        self.form_layout.addRow('Schema Version', self.ds_schema_version)
        self.form_layout.addRow('Type', self.ds_type)
        self.form_layout.addRow('Volume', self.ds_volume)
        self.form_layout.addRow('PPM', self.ds_ppm)
        self.form_layout.addRow('Invert Normals', self.ds_invert_normals)
        self.form_layout.addRow('Mask', self.ds_mask)
        self.form_layout.addRow('Ink Label', self.ds_ink_label)
        self.form_layout.addRow('RGB Label', self.ds_rgb_label)
        self.form_layout.addRow('VC Texture Label', self.ds_vct_label)
        self.form_layout.addRow('Bounding Box', self.ds_bounding_box)

        self.save_btn = QPushButton('Save')
        self.save_btn.clicked.connect(self.save)

        v_layout = QVBoxLayout()
        v_layout.addLayout(self.form_layout)
        v_layout.addWidget(self.save_btn)

        self.update_fields()
        self.setLayout(v_layout)

    def launch_bounds_editor(self):
        self.ds_bounding_box.launch_bounds_editor()

    def update_fields(self):
        region = True if self._datasource.getType() == 'region' else False
        self.ds_ppm.setEnabled(region)
        self.ds_mask.setEnabled(region)
        self.ds_rgb_label.setEnabled(region)
        self.ds_ink_label.setEnabled(region)
        self.ds_invert_normals.setEnabled(region)
        self.ds_vct_label.setEnabled(region)
        self.ds_bounding_box.setEnabled(region)
        self.save_btn.setEnabled(self._tainted)

    @Slot(bool)
    def save(self, checked: bool = False):
        self._datasource.save()
        self._tainted = False
        self.update_fields()

    @Slot(int)
    def update_invert_normals(self, state: int):
        self._datasource.setInvertNormals(state > 0)
        self._tainted = True
        self.update_fields()

    @Slot(str)
    def update_type(self, value: str):
        self._datasource.setType(value)
        self._tainted = True
        self.update_fields()

    @Slot(str)
    def update_ppm(self, value: str):
        self._datasource.setPPM(value)
        self.ds_bounding_box.ppm_changed()
        self._tainted = True
        self.update_fields()

    @Slot(str)
    def update_volume(self, value: str):
        self._datasource.setVolume(value)
        self._tainted = True
        self.update_fields()

    @Slot(str)
    def update_mask(self, value: str):
        self._datasource.setMask(value)
        self._tainted = True
        self.update_fields()

    @Slot(str)
    def update_ink_label(self, value: str):
        self._datasource.setInkLabel(value)
        self._tainted = True
        self.update_fields()

    @Slot(str)
    def update_rgb_label(self, value: str):
        self._datasource.setRGBLabel(value)
        self._tainted = True
        self.update_fields()

    @Slot(str)
    def update_vct_label(self, value: str):
        self._datasource.setVCTLabel(value)
        self._tainted = True
        self.update_fields()

    @Slot(list)
    def update_bounding_box(self, value: list):
        self._datasource.setBoundingBox(value)
        self._tainted = True
        self.update_fields()

    @Slot(str)
    def switch_files(self, filename: str):
        if self._tainted:
            save_yorn = QMessageBox.question(self, 'Save changes?',
                                             'You are about to edit a different file, but you have unsaved changes. Would you like to save these changes before leaving?')
            if save_yorn == QMessageBox.Yes:
                self.save()
        self.switch_editors.emit(filename)

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
        return DatasourceEditor(self.path, self.model().datasources(), parent=parent)


class DatasetModel(QStandardItemModel):
    def __init__(self, filename: str, parent: Optional[QObject]):
        super().__init__(parent)
        self._path = Path(filename)
        self._seen = []
        self._datasources = []
        self._datasource_index_map = {}
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
            self._datasources.append(str(path))
            self._datasource_index_map[str(path)] = datasource_item.index()
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

    def path_to_index(self, path: str):
        if path not in self._datasource_index_map:
            return None
        return self._datasource_index_map[path]

    def path(self):
        return self._path

    def datasources(self):
        return self._datasources
