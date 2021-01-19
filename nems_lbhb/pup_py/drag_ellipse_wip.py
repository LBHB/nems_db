import PIL.Image
from PIL import ImageTk
from tkinter import *

class ExampleApp(Frame):
    def __init__(self,master):
        Frame.__init__(self,master=None)
        self.x = self.y = 0
        self.canvas = Canvas(master,  cursor="cross")

        self.sbarv=Scrollbar(self,orient=VERTICAL)
        self.sbarh=Scrollbar(self,orient=HORIZONTAL)
        self.sbarv.config(command=self.canvas.yview)
        self.sbarh.config(command=self.canvas.xview)

        self.canvas.config(yscrollcommand=self.sbarv.set)
        self.canvas.config(xscrollcommand=self.sbarh.set)

        self.canvas.grid(row=0,column=0,sticky=N+S+E+W)
        self.sbarv.grid(row=0,column=1,stick=N+S)
        self.sbarh.grid(row=1,column=0,sticky=E+W)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # add rotation bindings
        self.canvas.bind("<Button-3>", self.on_rotate_press)
        self.canvas.bind("<B3-Motion>", self.on_rotate_motion)

        self.rect = None

        self.start_x = None
        self.start_y = None

        self.im = PIL.Image.open("/auto/users/hellerc/code/projects/in_progress/TIN_behavior/dump_figs/CRD010b/PCA_space.png")
        self.wazil,self.lard=self.im.size
        self.canvas.config(scrollregion=(0,0,self.wazil,self.lard))
        self.tk_im = ImageTk.PhotoImage(self.im)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)   

    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y

        # create rectangle if not yet exist
        #if not self.rect:
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_oval(self.x, self.y, 1, 1, fill="")

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)    
        #self.canvas.coords(self.rect, tuple(poly_oval(self.start_x, self.start_y, curX, curY, rotation=10))) 

    def on_button_release(self, event):
        pass    
    
    def getangle(self, event):
        dx = self.canvas.canvasx(event.x) - self.center[0]
        dy = self.canvas.canvasy(event.y) - self.center[1]
        try:
            return complex(dx, dy) / abs(complex(dx, dy))
        except ZeroDivisionError:
            return 0.0 # cannot determine angle

    def on_rotate_press(self, event):
        # calculate angle at start point
        self.center = ((self.canvas.coords(self.rect)[0] + self.canvas.coords(self.rect)[1]) / 2),  \
                            ((self.canvas.coords(self.rect)[2] + self.canvas.coords(self.rect)[3]) / 2)
        self.xy = [
            (self.canvas.coords(self.rect)[0], self.canvas.coords(self.rect)[2]),
            (self.canvas.coords(self.rect)[1], self.canvas.coords(self.rect)[2]),
            (self.canvas.coords(self.rect)[1], self.canvas.coords(self.rect)[3]),
            (self.canvas.coords(self.rect)[0], self.canvas.coords(self.rect)[3]),
        ]
        self.start = self.getangle(event)

    def on_rotate_motion(self, event):
        # calculate current angle relative to initial angle
        angle = self.getangle(event) / self.start
        offset = complex(self.center[0], self.center[1])
        newxy = []
        for x, y in self.xy:
            v = angle * (complex(x, y) - offset) + offset
            newxy.append((v.real, v.imag))
        self.canvas.coords(self.rect, *newxy)

if __name__ == "__main__":
    root=Tk()
    app = ExampleApp(root)
    root.mainloop()