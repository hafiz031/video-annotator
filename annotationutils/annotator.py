import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

class Annotator:
    def __init__(self, file_dir, output_dir):
        self.file_dir = file_dir
        self.output_dir = output_dir

    
    def on_press(self, event):
        print('press', event.key)
        # sys.stdout.flush()
        if event.key == 'x':
            # visible = xl.get_visible()
            # xl.set_visible(not visible)
            # fig.canvas.draw()
            print("FFF")


    def annotate(self):
        ## from vide
        cap = cv.VideoCapture(self.file_dir)
            
        ck = 0
        cnt = 0

        # fig, ax = plt.subplots()
        # print(fig.figure)
        while True:
            print("BBB")
            ret, img = cap.read()
            if not ret:
                break

            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            # #####################################   
            # print(img)
            # # plt.imshow(x)
            # plt.figure(1)
            # plt.clf() # clear the current figure
            # plt.imshow(img)
            # # plt.title('Number ' + str(cnt))
            # plt.pause(0.001)
            # # while True:
            # #     plt.pause(0.001)
            # #     k = cv.waitKey(0)
            # #     if k == 27:
            # #         break
            # #     print("CCC")
            # # print("DDD")
            # cnt += 1
            # #####################################

            #####################################
            # plt.axis('equal')
            # plt.show()
            
            # fig.show()

            # break
            # cv.imshow('image',img)
            # cv.waitKey(20)
            # while(1):
            #     _img = img.copy()
            #     cv.imshow('image',_img)
            #     k = cv.waitKey(20) & 0xFF
            #     if k == 27:  # skip frame
            #         cnt = cnt + 1
            #         cv.destroyAllWindows()
            #         break
                    
            #     elif k == ord('x'): # terminate
            #         ck = 1
            #         cv.destroyAllWindows()
            #         break
            #     # cv.imwrite('annotated_frames_'+video_title+'/'+str(cnt)+".jpg",_img)
            #####################################

            fig, ax = plt.subplots()

            fig.canvas.mpl_connect('key_press_event', self.on_press)
            plt.imshow(img)
            # ax.plot()
            # xl = ax.set_xlabel('easy come, easy go')
            # ax.set_title('Press a key')
            plt.show()
            plt.clf()
                

annotator = Annotator(file_dir="/media/hafiz031/HDD_Partition_2/Thesis/may2022/may0222/hafiz.mp4",
                    output_dir = '.')
annotator.annotate()



# import numpy as np
# from matplotlib import pyplot as plt

# for j in range(0,3):
#     img = np.random.normal(size=(100,150))
#     plt.figure(1); 
#     # plt.clf()
#     plt.imshow(img)
#     plt.title('Number ' + str(j))
#     plt.pause(3)