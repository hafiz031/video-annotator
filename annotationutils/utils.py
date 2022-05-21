# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# class DraggableRectangle:
#     def __init__(self, rect):
#         self.rect = rect
#         self.press = None

#     def connect(self):
#         """Connect to all the events we need."""
#         self.cidpress = self.rect.figure.canvas.mpl_connect(
#             'button_press_event', self.on_press)
#         self.cidrelease = self.rect.figure.canvas.mpl_connect(
#             'button_release_event', self.on_release)
#         self.cidmotion = self.rect.figure.canvas.mpl_connect(
#             'motion_notify_event', self.on_motion)

#     def on_press(self, event):
#         """Check whether mouse is over us; if so, store some data."""
#         if event.inaxes != self.rect.axes:
#             return
#         contains, attrd = self.rect.contains(event)
#         if not contains:
#             return
#         print('event contains', self.rect.xy)
#         print(event.xdata) # from where dragging started
#         print(event.ydata)
#         # self.rect.xy is the (left, bottom) = (x, y) of the object
#         # (event.xdata, event.ydata)is the position just where mouse click happened on that object
#         # so I may save the last drawn rectangle's (right, top) such that I can remember the rectangle's shape for 
#         # changing situation (after dragging)
#         self.press = self.rect.xy, (event.xdata, event.ydata) # drag won't happen without it

#     def on_motion(self, event):
#         """Move the rectangle if the mouse is over us."""
#         if self.press is None or event.inaxes != self.rect.axes:
#             return
#         (x0, y0), (xpress, ypress) = self.press
#         dx = event.xdata - xpress
#         dy = event.ydata - ypress
#         print(f'x0={x0}, xpress={xpress}, event.xdata={event.xdata}, '
#               f'dx={dx}, x0+dx={x0+dx}')
#         # The following two lines are basically drawing the rectangle while dragging
#         self.rect.set_x(x0+dx)
#         self.rect.set_y(y0+dy)

#         self.rect.figure.canvas.draw() # showing the object while dragging happens here

#     def on_release(self, event): 
#         """Clear button press information."""
#         self.press = None # without it you won't be able to release the object from the cursor once you clicked on it
#         self.rect.figure.canvas.draw()

#     def disconnect(self):
#         """Disconnect all callbacks."""
#         self.rect.figure.canvas.mpl_disconnect(self.cidpress)
#         self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
#         self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

# fig, ax = plt.subplots()

# rects = ax.bar(range(10), 20*np.random.rand(10)) # These are the rectangle4
# print(list(rects))

# drs = []
# for rect in rects:
#     dr = DraggableRectangle(rect)
#     dr.connect()
#     drs.append(dr)

# plt.show()




# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# from matplotlib.widgets  import RectangleSelector

# xdata = np.linspace(0, 10, num = 100)
# ydata = np.linspace(0, 10, num = 100)

# fig, ax = plt.subplots()
# line, = ax.plot(xdata, ydata)


# def line_select_callback(eclick, erelease):
#     x1, y1 = eclick.xdata, eclick.ydata
#     x2, y2 = erelease.xdata, erelease.ydata
#     print(eclick.x)
#     print(eclick.y)
#     print(dir(eclick))
#     print(eclick.lastevent) # we can see the xy and find if from which direction the dragging happened
#     print(x1, y1) # (left, bottom)
#     print(x2, y2) # (right, top)
#     print("#")
#     rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
#     ax.add_patch(rect)



# rs = RectangleSelector(ax, line_select_callback,
#                        drawtype='box', useblit=False, button=[1], 
#                        minspanx=5, minspany=5, spancoords='pixels', 
#                        interactive=True)

# plt.show()



import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.widgets  import RectangleSelector
import config


# x = np.array(Image.open('annotationutils/black.jpg'), dtype=np.uint8)
# # plt.imshow(x)
# fig, ax = plt.subplots(1)
# ax.imshow(x)

class RegionOfInterestDrawingUtils:
    def __inti__(self):
        pass


    def find_dragging_alignment(self, left_top, right_bottom, mouse_release_point):
        """There are 4 cases there:
        1. Dragged from left-top
        2. Dragged from right-bottom
        3. Dragged from right-top
        4. Dragged from left-bottom
        """

        # To get rid of the floating point precision error in further calculation
        left_top = [int(v) for v in left_top]
        right_bottom = [int(v) for v in right_bottom]
        mouse_release_point = [int(v) for v in mouse_release_point]
        
        x1, y1 = left_top
        x2, y2 = right_bottom
        x, y = mouse_release_point

        print(f"left_top: {left_top}")
        print(f"right_bottom: {right_bottom}")
        print(f"mouse_release_point: {mouse_release_point}")

        expression = (x - x1) * (y1 - y2) - (y - y1) * (x1 - x2)

        if math.isclose(expression, 0):
            if left_top == mouse_release_point:
                return "towards_left_top"
            elif right_bottom == mouse_release_point:
                return "towards_right_bottom"
        elif expression > 0:
            return "towards_left_bottom"
        elif expression < 0:
            return "towards_right_top"


    def rectangle_drawing_callback(self, eclick, erelease, make_square = False):
        """All measurements are being taken considering (left, top) = (0, 0)"""
        left_top = eclick.xdata, eclick.ydata
        right_bottom = erelease.xdata, erelease.ydata
        mouse_release_point = eclick.lastevent.xdata, eclick.lastevent.ydata 


        x1, y1 = left_top
        x2, y2 = right_bottom

        if make_square:
            """There are 4 possibilities (not considering the cases when user drags along
            a straight-line, as this won't create a rectangle/square). Among them we need not
            handle the case when mouse is dragged from left-top. But the other 3 cases need to
            be handled by introducing offset in order to give a natural feeling while drawing.
            Among those 3 cases, there are only 4 sub-cases in total where we need to do this.
            Otherwise the resulting square won't be set at the proper position."""

            min_dist = min(np.abs(x1-x2), np.abs(y1-y2))
            
            dragging_alignment = self.find_dragging_alignment(left_top, right_bottom, mouse_release_point)

            if dragging_alignment == "towards_left_bottom":
                if abs(y2 - y1) < abs(x2 - x1):
                    x1 = x2 - min_dist
            elif dragging_alignment == "towards_left_top":
                if abs(y2 - y1) < abs(x2 - x1):
                    x1 = x2 - min_dist
                else:
                    y1 = y2 - min_dist
            elif dragging_alignment == "towards_right_top":
                if abs(y2 - y1) > abs(x2 - x1):
                    y1 = y2 - min_dist

            rect = plt.Rectangle((min(x1,x2),min(y1,y2)), min_dist, min_dist)
        else:
            rect = plt.Rectangle((min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2))

        ax.add_patch(rect)


        def draw_roi(self):
            roi_draw = RegionOfInterestDrawingUtils()

            if config.ROI_TYPE == "square":
                rs = RectangleSelector(ax, lambda eclick, erelease: roi_draw.rectangle_drawing_callback(eclick, erelease, make_square = True),
                                    drawtype='box', useblit=False, button=[1], # usebilt True means the shape will not be shown
                                    minspanx=5, minspany=5, spancoords='pixels', 
                                    interactive=False)
            elif config.ROI_TYPE == "rectangle":
                rs = RectangleSelector(ax, lambda eclick, erelease: roi_draw.rectangle_drawing_callback(eclick, erelease, make_square = False),
                                    drawtype='box', useblit=False, button=[1], # usebilt True means the shape will not be shown
                                    minspanx=5, minspany=5, spancoords='pixels', 
                                    interactive=False)


plt.axis('equal')
plt.show()

