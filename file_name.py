import os

ori_dir = '/home/chj/u2net_data/original'
mask_dir = '/home/chj/u2net_data/original_mask'

original_file = os.listdir(ori_dir)
mask_file = os.listdir(mask_dir)

def file_name(file_list):
    
    file_name = []
    for file in file_list:
        if file.count(".") == 1: 
            name = file.split('.')[0]
            file_name.append(name)
        else:
            for k in range(len(file)-1,0,-1):
                if file[k]=='.':
                    file_name.append(file[:k])
                    break
    file_name
    

ori_file_name = file_name(original_file)
mask_file_name = file_name(mask_file)

compare_name = list(set(ori_file_name) - set(mask_file_name))

with open('/home/chj/u2net_data/original/file_name.txt','w',encoding='UTF-8') as f:
    for name in compare_name:
        f.write(name+'\n')