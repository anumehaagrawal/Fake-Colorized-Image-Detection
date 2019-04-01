import cv2
from matplotlib import pyplot as plt
import os
import pickle

def padImages(folder,img_type):
    images = []
    names = []
    max_h=-100
    max_w=-100

    img_count=0
    for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,fname))
            if(len(img)>max_h):
                max_h=len(img)
            if(len(img[0])>max_w):
                max_w=len(img[0])

    for fname in os.listdir(folder):
        img_count+=1
        img = cv2.imread(os.path.join(folder,fname))
        h=len(img)
        w=len(img[0])
        top=left=right=bottom=0
        top=(max_h-h)//2
        bottom=(max_h-h)%2+top
        left=(max_w-w)//2
        right=(max_w-w)%2+left
        padded_img= cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[255,255,255])
        print("Dim ",len(padded_img),len(padded_img[0]))
        cv2.imwrite(img_type+"/img"+str(img_count)+".jpg",padded_img)


def inImage(img,i,j):
    if(i>=0 and i<len(img) and j>=0 and j<len(img[0])):
        return True
    return False

def rotated(array_2d):
    list_of_tuples = zip(*array_2d[::-1])
    return [list(elem) for elem in list_of_tuples]
    # return map(list, list_of_tuples)

def gmm_get_phi_preprocess(images,img_type):
    phi=[]
    # file=open(img_type+".txt","w")
    imgc=0
    for img in images:
        # r1=[[1,2,3],[4,5,6],[3,1,2],[6,2,1],[4,5,6],[3,1,2],[6,2,1]]
        # r2=[[7,8,9],[10,1,2],[10,6,7],[8,11,5],[4,5,6],[3,1,2],[6,2,1]]
        # r3=[[1,0,7],[3,6,9],[8,4,3],[2,9,3],[4,5,6],[3,1,2],[6,2,1]]
        # r4=[[4,6,7],[8,1,2],[0,2,1],[2,0,9],[4,5,6],[3,1,2],[6,2,1]]
        # img=[r1,r2,r3,r4]
        imgc+=1
        h_vec=[]
        s_vec=[]
        bc_vec=[]
        dc_vec=[]
        hsv_mat=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gbr_mat=img
        # hsv_mat=[r1,r2,r3,r4]
        # gbr_mat=[r1,r2,r3,r4]
        #Method 1 - Use all pixel values
        #Extract Ih, Is
        # for row in hsv_mat:
        #     h_temp=[]
        #     s_temp=[]
        #     for hsv_trip in row:
        #         h_temp.append(hsv_trip[0])
        #         s_temp.append(hsv_trip[1])
        #     h_vec.append(h_temp)
        #     s_vec.append(s_temp)

        # #Extract Ibc, Idc
        # radius=1
        # for row in range(len(gbr_mat)):
        #     dc_temp=[]
        #     bc_temp=[]
        #     for col in range(len(gbr_mat[0])):
        #         min_dc=1000
        #         max_bc=-1000
        #         for r_row in range(-radius,radius+1):
        #             for r_col in range(-radius,radius+1):
        #                 # a=1
        #                 if(inImage(gbr_mat,row+r_row,col+r_col)):
        #                     # a=1
        #                     print(row+r_row,col+r_col)
        #                     rgb_min=min(gbr_mat[row+r_row][col+r_col][0],min(gbr_mat[row+r_row][col+r_col][1],gbr_mat[row+r_row][col+r_col][2]))
        #                     if(min_dc>rgb_min):
        #                         min_dc=rgb_min
        #                     rgb_max=max(gbr_mat[row+r_row][col+r_col][0],max(gbr_mat[row+r_row][col+r_col][1],gbr_mat[row+r_row][col+r_col][2]))
        #                     if(max_bc<rgb_max):
        #                         max_bc=rgb_max
        
        #         dc_temp.append(min_dc)
        #         bc_temp.append(max_bc)
        #     bc_vec.append(bc_temp)
        #     dc_vec.append(dc_temp)

        #Method 2 By considering averages over windows
        wind_size=10
        #Extract Ih, Is
        for row in range(0,len(hsv_mat),wind_size):
            h_temp=[]
            s_temp=[]
            for col in range(0,len(hsv_mat[0]),wind_size):
                avg_h=avg_s=0
                count=0
                for i in range(wind_size):
                    for j in range(wind_size):
                        if(inImage(hsv_mat,row+i,col+j)):
                            count+=1
                            avg_h+=(hsv_mat[row+i][col+j][0])
                            avg_s+=(hsv_mat[row+i][col+j][1])
                h_temp.append(avg_h//count)
                s_temp.append(avg_s//count)
            h_vec.append(h_temp)
            s_vec.append(s_temp)
        #Extract Ibc, Idc
        radius=1
        for row_ind in range(0,len(gbr_mat),wind_size):
            dc_temp=[]
            bc_temp=[]
            for col_ind in range(0,len(gbr_mat[0]),wind_size):
                avg_bc=avg_dc=0
                count=0
                for i in range(wind_size):
                    for j in range(wind_size):
                        if(inImage(gbr_mat,row_ind+i,col_ind+j)):
                            count+=1
                            min_dc=1000
                            max_bc=-1000
                            row=row_ind+i
                            col=col_ind+j
                            count2=0
                            for r_row in range(-radius,radius+1):
                                for r_col in range(-radius,radius+1):
                                    if(inImage(gbr_mat,row+r_row,col+r_col)):
                                        count2+=1
                                        rgb_min=min(gbr_mat[row+r_row][col+r_col][0],min(gbr_mat[row+r_row][col+r_col][1],gbr_mat[row+r_row][col+r_col][2]))
                                        if(min_dc>rgb_min):
                                            min_dc=rgb_min
                                        rgb_max=max(gbr_mat[row+r_row][col+r_col][0],max(gbr_mat[row+r_row][col+r_col][1],gbr_mat[row+r_row][col+r_col][2]))
                                        if(max_bc<rgb_max):
                                            max_bc=rgb_max
                            # print("Count2:",count2)
                            avg_bc+=(max_bc)
                            avg_dc+=(min_dc)
                dc_temp.append(avg_bc//count)
                bc_temp.append(avg_dc//count)
            bc_vec.append(bc_temp)
            dc_vec.append(dc_temp)
        
        # print(h_vec)
        h_vec=sum(h_vec,[])
        s_vec=sum(s_vec,[])

        # print(dc_vec)
        bc_vec=sum(bc_vec,[])
        dc_vec=sum(dc_vec,[]) 
        feature_vec=list(rotated([h_vec,s_vec,bc_vec,dc_vec]))
        phi.append(feature_vec)
        print("Image No. ",imgc)

        # file.write(','.join(str(e) for e in h_vec))
        # file.write(','.join(str(e) for e in s_vec))
        # file.write(','.join(str(e) for e in bc_vec))
        # file.write(','.join(str(e) for e in dc_vec))
        # emp = {1:"A",2:"B",3:"C",4:"D",5:"E"}
    pickling_on = open("phi_"+img_type+".pickle","wb")
    pickle.dump(phi, pickling_on)
    print(len(phi),len(phi[0]),len(phi[0][0]))
    pickling_on.close()


#Loading images and their names into an array
def load_images_from_folder(folder):
    images = []
    names = []
    for fname in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,fname))
            # print(len(img),len(img[0]))
            if img is not None:
                    images.append(img)
                    names.append(fname)
    return images,names

#padImages("sun6-gthist","fake_padded")
images,names=load_images_from_folder("true_padded")
# gmm_get_phi_preprocess(images,"true")