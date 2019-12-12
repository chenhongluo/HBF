import numpy as np
f1Size=[]
times=[]
warps=[]
with open('log.txt') as f:
    while(1):
        line = f.readline().strip('\n')
        nums1 = line.split(' ')
        if(len(nums1) == 1):
            break
        f1Size.append(int(nums1[0]))
        times.append(int(nums1[1]))
        line = f.readline().strip('\n')
        nums2 = line.split(' ')[:-1]
        temp=[]
        for x in nums2:
            temp.append(int(x))
        warps.append(temp)

    fn = np.array(f1Size)
    tn = np.array(times)
    wn = np.array(warps)

    tnall = np.sum(tn)
    tnp = tn / tnall

    wna = np.average(wn,1)

    wnm = np.max(wn,1)
    wnp = wna/wnm

    tnn = tn * wnp
    tnnall = np.sum(tnn)

    for i in range(len(fn)):
        print(fn[i],tn[i],wnp[i])


    print(tnall,tnnall,(tnall-tnnall)/tnall)