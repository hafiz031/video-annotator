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





import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.widgets  import RectangleSelector


x = np.array(Image.open('annotationutils/black.jpg'), dtype=np.uint8)
# plt.imshow(x)
fig, ax = plt.subplots(1)
ax.imshow(x)


def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(eclick.x)
    print(eclick.y)
    print(dir(eclick))
    print(eclick.lastevent) # we can see the xy and find if from which direction the dragging happened
    print(x1, y1) # (left, bottom)
    print(x2, y2) # (right, top)
    print("#")
    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    ax.add_patch(rect)



rs = RectangleSelector(ax, line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)

plt.axis('equal')
plt.show()

