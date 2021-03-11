import os
import numpy as np
# list all classes from ImageNet 2011 database which have enough images
large_imagenet=os.listdir('/mnt/fast-data15/datasets/imagenet/fa2011')
os.chdir('/mnt/fast-data15/datasets/imagenet/fa2011')
list_of_classes=[]
for n in range(len(large_imagenet)):
	foldername=large_imagenet[n]
	if len(os.listdir(foldername))>=1350:
		list_of_classes.append(foldername)
small_imagenet=os.listdir('/home/daniel/0_imagenet_split/train')
intersection_set = set.intersection(set(list_of_classes), set(small_imagenet))
# choose only classes not in the smaller Imagenet dataset of 1000 classes
unique_large_imagenet = list(set(list_of_classes) - set(small_imagenet))
import shutil
# choose 150 classes at random, to be reduced to 100 following manual inspection of existing
# similarity to the smaller ImageNet dataset
for n in range(150):
	from_folder=os.path.join('/mnt/fast-data15/datasets/imagenet/fa2011',unique_large_imagenet[n])
	to_folder=os.path.join('/home/daniel/replay/imagenet',unique_large_imagenet[n])
	shutil.copytree(from_folder,to_folder)
labels=np.genfromtxt('/home/daniel/replay/fa11_labels.csv',delimiter=',',dtype=None)
class_ids=[]
for m in range(150):
	arr_index = np.where(labels == unique_large_imagenet[m])
	class_ids.append(labels[arr_index[0][0]])
np.savetxt('/home/daniel/replay/class_ids.csv',class_ids_np,delimiter=" ",fmt="%s")
# randomly assign the 100 classes to 10 different models
class_list=os.listdir("/home/daniel/replay/imagenet")
print(class_list)
os.chdir('/home/daniel/replay/class_lists')
np.save('/home/daniel/replay/class_lists/all_classes.npy',class_list)
for n in range(10):
	start=n*10
	end=(n*10)+10
	print('start',start)
	print('end',end)
	temp_class_list=class_list[(n*10):((n*10)+10)]
	print(temp_class_list)
	filename=str(n)+'.npy'
	np.save(filename,temp_class_list)