import os
import shutil

dir = "ctest10k"
for path in os.listdir(dir):
    full_path = os.path.join(dir, path)
    print(full_path)
    name = os.path.basename(path)
    shutil.move("ILSVRC2012_img_val/"+name,"original_images"+name)

