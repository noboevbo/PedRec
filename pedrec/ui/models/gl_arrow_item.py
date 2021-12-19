import math

import OpenGL.GL as ogl
import OpenGL.GLU as glu
import pyqtgraph.opengl as gl

from pedrec.models.data_structures import Color


class GLArrowItem(gl.GLGraphicsItem.GLGraphicsItem):
    """
    Based on C++ implementation of The Qunatum Physicist
    @ https://stackoverflow.com/questions/19332668/drawing-the-axis-with-its-arrow-using-opengl-in-visual-studio-2010-and-c
    """

    def __init__(self,
                 x1: float,
                 y1: float,
                 z1: float,
                 x2: float,
                 y2: float,
                 z2: float,
                 line_width: float = 1,
                 color: Color = Color(255, 255, 255, 255)):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2
        self.diameter = line_width * 0.01
        self.color = color.tuplef_rgba
        self.rad_per_deg = 0.0174533

    def paint(self):
        self.setupGLState()
        ogl.glColor4f(*self.color)
        x = self.x2 - self.x1
        y = self.y2 - self.y1
        z = self.z2 - self.z1
        L = math.sqrt(x * x + y * y + z * z)

        ogl.glPushMatrix()
        ogl.glTranslated(self.x1, self.y1, self.z1)

        if x != 0 or y != 0:
            ogl.glRotated(math.atan2(y, x) / self.rad_per_deg, 0, 0, 1)
            ogl.glRotated(math.atan2(math.sqrt(x * x + y * y), z) / self.rad_per_deg, 0, 1, 0)
        elif z < 0:
            ogl.glRotated(180, 1, 0, 0)

        ogl.glTranslatef(0, 0, L - 4 * self.diameter)

        # The arrow head
        quad_obj = glu.gluNewQuadric()
        glu.gluQuadricDrawStyle(quad_obj, glu.GLU_FILL)
        glu.gluQuadricNormals(quad_obj, glu.GLU_SMOOTH)
        glu.gluCylinder(quad_obj, 2 * self.diameter, 0.0, 4 * self.diameter, 32, 1)
        glu.gluDeleteQuadric(quad_obj)

        # Closing disk of the arrow head
        quad_obj = glu.gluNewQuadric()
        glu.gluQuadricDrawStyle(quad_obj, glu.GLU_FILL)
        glu.gluQuadricNormals(quad_obj, glu.GLU_SMOOTH)
        glu.gluDisk(quad_obj, 0.0, 2 * self.diameter, 32, 1)
        glu.gluDeleteQuadric(quad_obj)

        # The arrow shaft
        ogl.glTranslatef(0, 0, -L + 4 * self.diameter)
        quad_obj = glu.gluNewQuadric()
        glu.gluQuadricDrawStyle(quad_obj, glu.GLU_FILL)
        glu.gluQuadricNormals(quad_obj, glu.GLU_SMOOTH)
        glu.gluCylinder(quad_obj, self.diameter, self.diameter, L - 4 * self.diameter, 32, 1)
        glu.gluDeleteQuadric(quad_obj)

        # Closing disk at the origin of the arrow shaft
        quad_obj = glu.gluNewQuadric()
        glu.gluQuadricDrawStyle(quad_obj, glu.GLU_FILL)
        glu.gluQuadricNormals(quad_obj, glu.GLU_SMOOTH)
        glu.gluDisk(quad_obj, 0.0, self.diameter, 32, 1)
        glu.gluDeleteQuadric(quad_obj)

        ogl.glPopMatrix()
