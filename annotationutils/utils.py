import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DraggableRectangle:
    def __init__(self, rect):
        self.rect = rect
        self.press = None

    def connect(self):
        """Connect to all the events we need."""
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Check whether mouse is over us; if so, store some data."""
        if event.inaxes != self.rect.axes:
            return
        contains, attrd = self.rect.contains(event)
        if not contains:
            return
        print('event contains', self.rect.xy)
        print(event.xdata) # from where dragging started
        print(event.ydata)
        # self.rect.xy is the (left, bottom) = (x, y) of the object
        # (event.xdata, event.ydata)is the position just where mouse click happened on that object
        # so I may save the last drawn rectangle's (right, top) such that I can remember the rectangle's shape for 
        # changing situation (after dragging)
        self.press = self.rect.xy, (event.xdata, event.ydata) # drag won't happen without it

    def on_motion(self, event):
        """Move the rectangle if the mouse is over us."""
        if self.press is None or event.inaxes != self.rect.axes:
            return
        (x0, y0), (xpress, ypress) = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        print(f'x0={x0}, xpress={xpress}, event.xdata={event.xdata}, '
              f'dx={dx}, x0+dx={x0+dx}')
        # The following two lines are basically drawing the rectangle while dragging
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)

        self.rect.figure.canvas.draw() # showing the object while dragging happens here

    def on_release(self, event): 
        """Clear button press information."""
        self.press = None # without it you won't be able to release the object from the cursor once you clicked on it
        self.rect.figure.canvas.draw()

    def disconnect(self):
        """Disconnect all callbacks."""
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

fig, ax = plt.subplots()

rects = ax.bar(range(10), 20*np.random.rand(10)) # These are the rectangle4
print(list(rects))

drs = []
for rect in rects:
    dr = DraggableRectangle(rect)
    dr.connect()
    drs.append(dr)

plt.show()