import av
import matplotlib.pyplot as plt
import numpy as np

fn = '/auto/data/daq/Amanita/AMT039/AMT039d02_p_VOC.photo.avi'

video_container = av.open(fn)
video_stream = [s for s in video_container.streams][0]

# define an roi based on first frame
F_mag = []
for i, packet in enumerate(video_container.demux(video_stream)):
    if i%100 == 0:
        print("frame: {}".format(i))

    if i == 0:
        frame = packet.decode()[0]
        frame_ = np.asarray(frame.to_image().convert('LA'))
        plt.imshow(frame_[:, :, 0])
        print("Before closing image, locate the center of your ROI!")
        plt.show()
        x = int(input("x center: "))
        y = int(input("y center: "))

        # define roi:
        x_range = np.arange(x-2, x+2)
        y_range = np.arange(y-2, y+2)

        roi = frame_[:, :, 0][x_range, :][:, y_range]
        fmag = np.mean(roi)
        F_mag.append(fmag)

    else:
        try:
            frame = packet.decode()[0]
            frame_ = np.asarray(frame.to_image().convert('LA'))
            roi = frame_[:, :, 0][x_range, :][:, y_range]
            fmag = np.mean(roi)
            F_mag.append(fmag)
        except:
            print('end of file reached')


plt.figure()
plt.plot(F_mag)

plt.show()
