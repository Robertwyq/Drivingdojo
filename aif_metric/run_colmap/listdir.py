import sys
import os
metapath=sys.argv[1]
ith = int(sys.argv[2])
file_list=os.listdir(metapath)
file_list=[temp for temp in file_list if ((not '.pkl' in temp) and (not '.py' in temp) and (not '.bash' in temp) and (not 'trajs' in temp))]
file_list.sort()
file_list=[temp for ind,temp in enumerate(file_list) if ind%8==ith]
# file_list=[file_list[0]]
print(' '.join(file_list))