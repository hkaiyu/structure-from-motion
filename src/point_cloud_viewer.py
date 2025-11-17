import sys
import time
import threading
import numpy as np
from vispy import app, scene
# from vispy.scene.widgets import Console
from PyQt5 import QtWidgets, QtCore

#TODO: ideally, we could use the vispy console to log our data, but I need to figure out how to make the console
# resizable.
class PointCloudViewer(scene.SceneCanvas):
    def __init__(self, title, width=1024, height=768):
        super().__init__(keys="interactive", title=title,
                         size=(width, height), show=True)
        self.unfreeze()

        # Data
        self._lock = threading.Lock()
        self.points3d = np.empty((0, 3), np.float32)
        self.colors = np.empty((0, 3), np.float32)
        self.camera_exts = []
        self.camera_visuals = []
        self.centroid = np.array([0.0, 0.0, 0.0], np.float32)

        # Main Qt Window
        self.qt_window = QtWidgets.QMainWindow()
        self.qt_window.setWindowTitle(title)
        container = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Settings panel
        self.settings_panel = QtWidgets.QWidget()
        self.settings_panel.setFixedWidth(160)
        self.settings_panel.setStyleSheet(
            "background-color: rgb(45,45,45); color: white;"
        )
        settings_layout = QtWidgets.QVBoxLayout(self.settings_panel)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        title_label = QtWidgets.QLabel("Settings")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        settings_layout.addWidget(title_label)
        settings_layout.addSpacing(10)

        self.chk_frustums = QtWidgets.QCheckBox("Show Frustums")
        self.chk_frustums.setChecked(True)
        self.chk_frustums.stateChanged.connect(self._toggleFrustrums)
        settings_layout.addWidget(self.chk_frustums)

        settings_layout.addStretch()

        main_layout.addWidget(self.settings_panel)

        # view container
        view_container = QtWidgets.QWidget()
        view_layout = QtWidgets.QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.addWidget(self.native)
        main_layout.addWidget(view_container, stretch=1)
        self.qt_window.setCentralWidget(container)
        self.qt_window.resize(width, height)
        self.qt_window.show()

        # vispy scene
        self.view = self.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(up="+y", fov=45)
        self.view.camera.flip = (True, True, False)
        self.view.camera.distance = 10.0
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.markers.set_gl_state("translucent", depth_test=True, blend=True)
        self._timer = app.Timer(interval=1/30, connect=self.onTimer, start=True)
        self.events.close.connect(self._onClose)

        self.freeze()

    def updatePoints(self, pts, colors=None):
        if pts is None or len(pts) == 0:
            with self._lock:
                self.points3d = np.empty((0, 3), np.float32)
                self.colors = np.empty((0, 3), np.float32)
            return

        pts = np.asarray(pts, np.float32)

        if colors is None or len(colors) != len(pts):
            colors = np.full((pts.shape[0], 3), [0.8, 0.9, 1.0], np.float32)

        with self._lock:
            self.points3d = pts.copy()
            self.colors = colors.copy()

        extent = np.ptp(self.points3d, axis=0).max()
        self.view.camera.center = self.points3d.mean(axis=0)
        self.view.camera.distance = max(1.0, float(extent) * 1.1)

    def updateCameraPoses(self, cameras):
        """Cameras: list of dicts {R,t,K,name,color}."""
        if cameras is None:
            cameras = []

        with self._lock:
            self.camera_exts.clear()
            for cam in cameras:
                self.camera_exts.append({
                    "R": np.asarray(cam["R"]),
                    "t": np.asarray(cam["t"]).reshape(3, 1),
                    "K": np.asarray(cam["K"]),
                    "name": cam.get("name", "Camera"),
                    "color": cam.get("color", "orange")
                })

        self._rebuildCameraFrustrums()

    def _rebuildCameraFrustrums(self):
        for vis in self.camera_visuals:
            vis.parent = None
        self.camera_visuals.clear()

        if not self.chk_frustums.isChecked():
            return

        for cam in self.camera_exts:
            R, t, K = cam["R"], cam["t"], cam["K"]
            color = cam["color"]

            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            scale = 0.00075 * (2 * cx + 2 * cy)
            corners = np.array([
                [-cx / fx, -cy / fy, 1.0],
                [ cx / fx, -cy / fy, 1.0],
                [ cx / fx,  cy / fy, 1.0],
                [-cx / fx,  cy / fy, 1.0],
            ]) * scale

            C = (-R.T @ t).ravel()
            corners_world = (R.T @ corners.T + C[:, None]).T

            lines = []
            for i in range(4):
                j = (i + 1) % 4
                lines.append(np.stack([C, corners_world[i]]))
                lines.append(np.stack([corners_world[i], corners_world[j]]))

            vis = scene.visuals.Line(
                pos=np.vstack(lines),
                color=color,
                width=2.0,
                connect='segments',
                parent=self.view.scene,
            )
            self.camera_visuals.append(vis)

    def _toggleFrustrums(self, state):
        visible = (state == QtCore.Qt.Checked)
        for vis in self.camera_visuals:
            vis.visible = visible

    def onTimer(self, event):
        with self._lock:
            pts = self.points3d.copy()
            cols = self.colors.copy()

        if pts.size > 0:
            if cols.shape[0] != pts.shape[0]:
                cols = np.full((pts.shape[0], 3),
                               [0.8, 0.9, 1.0],
                               dtype=np.float32)

            alpha = np.full((cols.shape[0], 1), 1.0, dtype=np.float32)
            rgba = np.concatenate([cols, alpha], axis=1).astype(np.float32)
            self.markers.set_data(
                pts,
                face_color=rgba,
                size=5.0,
                edge_width=0
            )
        else:
            self.markers.set_data(np.empty((0, 3), np.float32))

    def _onClose(self, event):
        app.quit()

    def run(self):
        app.run()
