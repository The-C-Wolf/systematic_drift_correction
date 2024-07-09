import os
import matplotlib.pyplot as plt
from PIL import Image

def show_trial(orig_XY, corrected_XY, subject, stimfile, eye, stimfiles_folder):
    
    img_fpath = os.path.join(stimfiles_folder, stimfile)
    img = Image.open(img_fpath)
    plt.figure(figsize=(10, 6))

    plt.imshow(img)

    #lines between consecutive points
    plt.plot(corrected_XY[:,0], corrected_XY[:,1], color="grey", alpha=0.3, linestyle='-', linewidth=1)

    plt.scatter(corrected_XY[:,0], corrected_XY[:,1], color="red", label="Corrected", alpha=0.6, s=20)
    
    for i in range(corrected_XY.shape[0]):
        orig_row = orig_XY[i]
        dx = corrected_XY[i, 0] - orig_row[0]
        dy = corrected_XY[i, 1] - orig_row[1]
        
        plt.arrow(orig_row[0], orig_row[1], dx, dy, 
                head_width=5, head_length=5, fc="black", ec="black", alpha=0.3, length_includes_head=True)


    plt.legend()
    plt.suptitle(f"{subject} {stimfile} (eye: {eye})")
    plt.show()