import meshcat
from meshcat import Visualizer
import meshcat.geometry as mc_geom
import meshcat.transformations as mc_trans
import numpy as np


class Arm(Visualizer):
    def __init__(self, l1, l2, m1, m2) -> None:
        Visualizer.__init__(self)

        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

        self._link1 = self['link1']
        self._mass1 = self._link1["mass1"]
        self._link2 = self._mass1["link2"]
        self._mass2 = self._link2["mass2"]

        self._mass1.set_object(
            mc_geom.Sphere(radius=0.1),
            mc_geom.MeshLambertMaterial(
                    # color=0x00ff00,
                    # opacity=0.5,
                    reflectivity=0.8,
                    )
        )
        self._mass2.set_object(
            mc_geom.Sphere(radius=0.1),
            mc_geom.MeshLambertMaterial(
                    # color=0x00ff00,
                    # opacity=0.5,
                    reflectivity=0.8,
                    )
        )

        self._link1.set_object(mc_geom.LineSegments(
            mc_geom.PointsGeometry(position=np.array([
            [0, 0, 0], [0, 0, -l1]]).astype(np.float32).T,
            color=np.array([
            [1, 0, 0], [1, 0.6, 0],
            [0, 1, 0], [0.6, 1, 0],
            [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            ),
            mc_geom.LineBasicMaterial(vertexColors=True))
        )
        self._link2.set_object(mc_geom.LineSegments(
            mc_geom.PointsGeometry(position=np.array([
            [0, 0, 0], [0, 0, -l2]]).astype(np.float32).T,
            color=np.array([
            [1, 0, 0], [1, 0.6, 0],
            [0, 1, 0], [0.6, 1, 0],
            [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            ),
            mc_geom.LineBasicMaterial(vertexColors=True))
        )

        self.render([0,0])

    def render(self, q):

        print(q)

        self._link1.set_transform(
            mc_trans.compose_matrix(
                translate=[0,0,0], 
                angles=[q[0],0,0]
            )
        )
        self._mass1.set_transform(mc_trans.translation_matrix([0,0,-self.l1]))
        self._link2.set_transform(
            mc_trans.rotation_matrix(q[1], [0,0,-self.l2])
        )
        self._mass2.set_transform(mc_trans.translation_matrix([0,0,-self.l2]))