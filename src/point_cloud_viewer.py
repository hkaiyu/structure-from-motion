import time
import threading
import numpy as np
from vispy import app, scene
from PyQt5 import QtWidgets

class PointCloudViewer(scene.SceneCanvas):
    def __init__(self, title, width=1024, height=768):
        super().__init__(keys="interactive", title=title, size=(width, height), show=True)
        self.unfreeze()

        self.points = np.empty((0,3), np.float32)
        self.born = np.empty((0,), np.float64)
        self.fade_seconds = 2.0
        self.new_rgb = np.array([1.0,0.3,0.2])
        self.stable_rgb = np.array([0.8,0.9,1.0])
        self._lock = threading.Lock()

        self.cameras = {}  # name -> dict(R,t,K)
        self.centroid = (0, 0, 0)

        # ==== Qt config ====
        # TODO: make nicer layout + styling
        self.qt_window = QtWidgets.QMainWindow()
        self.qt_window.setWindowTitle(title)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.addItem("Default")  # default
        self.dropdown.currentIndexChanged.connect(self.onCameraSelected)

        layout.addWidget(self.dropdown)
        layout.addWidget(self.native)
        self.qt_window.setCentralWidget(central)
        self.qt_window.resize(width, height)
        self.qt_window.show()

        # ==== Vispy config =====
        self.view = self.central_widget.add_view()
        self._useDefaultTurntable()
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.markers.set_gl_state("translucent", depth_test=True, blend=True)
        self._timer = app.Timer(interval=1/60, connect=self.onTimer, start=True)
        self.events.close.connect(self._onClose)

        self.freeze()

    def _useDefaultTurntable(self):
        self.view.camera = scene.TurntableCamera(up="+y", fov=45)
        self.view.camera.center = self.centroid
        self.view.camera.flip = (True, True, False)
        if self.points.size > 0:
            self.view.camera.distance = np.ptp(self.points, axis=0).max()
        else:
            self.view.camera.distance = 10.0

        self.view.camera.azimuth = 0
        self.view.camera.elevation = 0

    def _onClose(self, event):
        app.quit()

    def _reorientCamera(self, R, t, K, center=None):
        pass

    def onCameraSelected(self, idx):
        name = self.dropdown.currentText()
        print(f"{name} selected.")

        camInfo = self.cameras.get(name)
        if camInfo is None:
            return

        # TODO: actually implement this
        return

    def addPoints(self, pts):
        if pts is None or len(pts)==0: return
        pts = np.asarray(pts, np.float32)
        now = time.time()
        with self._lock:
            n = pts.shape[0]
            self.points = np.vstack([self.points, pts])
            self.born = np.concatenate([self.born, np.full(n, now)])

        # Update camera center and distance
        if self.points.shape[0] > 0:
            self.centroid = tuple(self.points.mean(axis=0))
            self.view.camera.center = self.centroid

            max_extent = np.ptp(self.points, axis=0).max()
            self.view.camera.distance = max(1.0, max_extent * 1.1)

        self.update()

    def addCameraPose(self, R, t, K, scale=2.0, color='orange', name=None):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        img_w, img_h = 2 * cx, 2 * cy

        corners = np.array([
            [-cx/fx, -cy/fy, 1.0],
            [ cx/fx, -cy/fy, 1.0],
            [ cx/fx,  cy/fy, 1.0],
            [-cx/fx,  cy/fy, 1.0],
        ]) * scale

        C = -R.T @ t
        corners_world = (R.T @ corners.T + C).T

        # Draw lines for pyramid edges
        lines = []
        for i in range(4):
            j = (i + 1) % 4
            lines.append(np.stack([C.ravel(), corners_world[i]]))
            lines.append(np.stack([corners_world[i], corners_world[j]]))

        lines = np.concatenate(lines)
        scene.visuals.Line(pos=lines, color=color, width=2, connect='segments', parent=self.view.scene)

        if name is None:
            name = f"Camera {len(self.cameras)}"
        self.cameras[name] = dict(R=R, t=t, K=K)
        self.dropdown.addItem(name)

    def onTimer(self, event):
        with self._lock:
            pts = self.points.copy()
            born = self.born.copy()

        if pts.size == 0:
            self.update()
            return

        now = time.time()
        age = now - born
        t = np.clip(age / self.fade_seconds, 0.0, 1.0)[:, None]

        # Small animation to show new points being added vs. old points via simple linear interpolaton
        # TODO: we could track color from image pixels and make sure they remain the same in point cloud
        rgb = (1 - t) * np.array([1.0, 1.0, 1.0]) + t * self.stable_rgb # rgb white -> stable_rgb
        alpha = 0.9 - 0.4 * t  # alpha 0.9 -> 0.5
        size = 10.0 - 2.0 * t  # size 10 -> 8

        rgba = np.concatenate([rgb, alpha], axis=1).astype(np.float32)
        self.markers.set_data(
            pts,
            face_color=rgba,
            size=size.squeeze(),
            edge_width=0,
        )
        self.update()

    def run(self):
        """Run the GUI (must be on main thread). Blocks until window is closed."""
        app.run()


