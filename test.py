
#coding=utf-8
import sys 
# import numpy as np 

# oilFeePerKm=8/100*6.5

# def minL(mat,begin,end):
#     mat1=np.zeros(mat.shape)+10000000
#     L=mat.shape[0]
#     while(1000):
#         for i in range(L):

    


# for a in sys.stdin:
#     b = a.strip()
#     mylist=b.split(";")
#     data=[]
#     for lines in mylist:
#         values = list(map(int, lines.split(',')))
#         data.append(values)
#     #提取地名
#     pos=[]
#     for line in data:
#         if line[0] not in pos:   
#             pos.append(line[0])
#         if line[1] not in pos:   
#             pos.append(line[1])
#     pos.sort()
#     L=len(pos)

#     pos_name={}
#     for i in range(L):
#         pos_name[pos[i]]=i

#     disMat=np.zeros((L,L))-1
#     maxSpeed=np.zeros((L,L))-1
#     level=np.zeros((L,L))
#     fee=np.zeros((L,L))
#     time=np.zeros((L,L))
#     for i in range(L):
#         disMat[i,i]=0
#         maxSpeed[i,i]=0

#     for line in data:
#         r=pos_name[line[0]]
#         c=pos_name[line[1]]
#         dist=line[2]
#         speed=line[3]
#         distMat[r,c]=dist
#         distMat[c,r]=dist
#         maxSpeed[r,c]=speed
#         maxSpeed[c,r]=speed

#     for i in range(L):
#         for j in range(i+1,L):
#             f=oilFeePerKm*distMat[i,j]
#             if(maxSpeed[i,j]==100):
#                 f+=0.5*distMat[i,j]
#             fee[i,j]=int(f)
#             fee[j,i]=int(f)

#             t=float(distMat[i,j])/maxSpeed[i,j]
#             if(level[i,j]==1):
#                 t+=5
#             elif(level[i,j]==2):
#                 t+=10
#             elif(level[i,j]==3):
#                 t+=20
#             time[i,j]=t
#             time[j,i]=t
    

                
    
#     L1=len(data)
#     beginPos=pos_name[data[0][0]]
#     endPos=pos_name[data[L1-1][1]]




    
            


    



odd=[2]



def isOdd(n):
    L=len(odd)
    lastOdd=odd[L-1]
    if n in odd:
        return True
    if n<=lastOdd:
        return False

    mid=n//2
    addOdd(mid)
    isodd=True
    for divNum in odd:
        if n%divNum==0:
            isodd=False
            break
    return isodd

def addOdd(m):
    L=len(odd)
    lastOdd=odd[L-1]
    if(m<=lastOdd):
        return
    for x in range(lastOdd+1,m+1):
        isOdd=True
        for divNum in odd:
            if x%divNum==0:
                isOdd=False
                break
        if isOdd:
            odd.append(x)

# if __name__ == "__main__":
#     a = 4

#     mid=a//2+1
#     num=0
#     for m in range(2,mid):
#         n=a-m
#         if isOdd(m) and isOdd(n):
#             num+=1
#     print("{0}".format(num))

print('fff')
for line in sys.stdin:
    a = int(line.strip())
    if a==0:
        print("end")
        continue
    mid=a//2+1
    num=0
    for m in range(2,mid):
        n=a-m
        if isOdd(m) and isOdd(n):
            num+=1
    print("%d" % num)



'''
#coding=utf-8
import sys 
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))
'''

"""
import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ans = 0
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        for v in values:
            ans += v
    print(ans)
"""
    # L=len(line)
    # pt1=0
    # pt2=L-1
    # isOk=True
    # if(line[pt1]!=line[pt2]):
    #     isOk=False    
    # while(pt2-pt1>1):
    #     if(line[pt1]!=line[pt2]):
    #         isOk=False
    #         break
    #     else:
    #         pt1+=1
    #         pt2-=1
    # if isOk:
    #     print("Y")
    # else:
    #     print("N")