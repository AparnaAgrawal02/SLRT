# import numpy
# import pickle 
# import os
# import cv2

# video_list = []
# path = "../../ASL"
# x =0 
# for dir1 in os.listdir(path):
#     print(x)
#     for file in os.listdir(os.path.join(path, dir1)):
#         data = {}
#         data["video_file"]=os.path.join(path, dir1, file)
#         data["name"]=file.split(".")[0]
#         data["label"] = dir1
#         cap = cv2.VideoCapture(os.path.join(path, dir1, file))
#         data["seq_len"] =int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         video_list.append(data)
#         cap.release()
#         x+=1

# with open('train_asl.pkl', 'wb') as f:
#     pickle.dump(video_list, f)
# print("train_asl.pkl generated")


import numpy
import pickle 
import os
import cv2

video_list = []
dir1 = "../../cutted/1"
# path1 = "../../ASL1"
x =0 
with open("cutted.txt","w") as f:
    
    #for dir1 in os.listdir(path):
        # if not os.path.exists(os.path.join(path1, dir1)):
        #     os.makedirs(os.path.join(path1, dir1))
        #print(x)
    for file in os.listdir(dir1):
        data = {}
        data["video_file"]=os.path.join( dir1, file)
        f.write(os.path.join( dir1, file)+" "+dir1+"\n")
        data["name"]=dir1+"_"+file.split(".")[0]
        data["label"] = 'try'
        cap = cv2.VideoCapture(os.path.join( dir1, file))
        data["seq_len"] =int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_list.append(data)
        
        #resize the video to 256 x 256
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #out = cv2.VideoWriter(os.path.join(path1, dir1, file), fourcc, 20.0, (256, 256))
        # while(cap.isOpened()):
        #     ret, frame = cap.read()
        #     if ret==True:
        #         frame = cv2.resize(frame, (256, 256))
        #         out.write(frame)
        #     else:
        #         break
        cap.release()
        x+=1

    with open('train_cutted.pkl', 'wb') as f:
        pickle.dump(video_list, f)
    print("train_cutted.pkl generated")


