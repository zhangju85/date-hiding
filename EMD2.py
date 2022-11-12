
from PIL import Image
import numpy as np
import math
import time
import os #用于查找目录下的文件

#写入文件
def SaveResult(str):
#将str写入结果文件中
    try:
        fname = time.strftime("%Y%m%d", time.localtime())
        f2 = open('result' + fname + '.txt','a+')
        f2.read()
        f2.write('--------------------------------------------------')
        f2.write('\n')
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f2.write(timestr)
        f2.write('\n')
        f2.write(str)
        f2.write('\n')
    finally:
        if f2:
            f2.close()
    return 0

#PSNR
def PSNR(image_array1,image_array2):
    #输入为两个图像数组，一维，大小相同
    assert(np.size(image_array1) == np.size(image_array2))
    n = np.size(image_array1)
    assert(n > 0)
    MSE = 0.0
    for i in range(0,n):
        MSE+=math.pow(int(image_array1[i]) - int(image_array2[i]),2)
    MSE = MSE / n
    if MSE > 0:
        rtnPSNR = 10 * math.log10(255 * 255 / MSE)
    else:
        rtnPSNR = 100
    return rtnPSNR

def EMD_2006(image_array,secret_string,k,n,image_file_name=''):
    #image_array:输入的一维图像数组
    #n为一组像素的数量,我理解n只能取2，4,8,16等值，取其他值会导致嵌入的bit数不好确定
    #n=2
    assert(n == 2 or n == 4 or n == 8 or n == 16 or n == 32 or n == 64)
    moshu = 2 * n + 1  #模数的底
    #分成n个像素一组
    num_pixel_groups = math.ceil(image_array.size / n)
    pixels_group = np.zeros((num_pixel_groups,n))
    i = 0
    while (i < num_pixel_groups):
        for j in range(0,n):
            if(i * n + j < image_array.size):
                 pixels_group[i,j] = image_array[i * n + j]
        i = i + 1
    #每一个像素组计算出一个fG值
    fG_array = np.zeros(num_pixel_groups)
    for i in range(0,num_pixel_groups):
        fG = 0
        for j in range(0,n):
            fG += (j + 1) * pixels_group[i,j]
        fG_array[i] = fG % moshu
    #-----------------------------------------------------------------------------------
    #从待嵌入bit串数据中取出m个比特，作为一组。m=math.log((2*n),2),以2为底的对数
    m = int(math.log((2 * n),2))
    #分组
    num_secret_groups = math.ceil(secret_string.size / m)
    secret_group = np.zeros((num_secret_groups,m))
    i = 0
    while (i < num_secret_groups):
        for j in range(0,m):
            if(i * m + j < s_data.size):#先判断是否在秘密信息范围内
                 secret_group[i,j] = s_data[i * m + j]
        i = i + 1
    #-----------------------------------------------------------------------------------

    #一组pixels_group嵌入一组secret_group的信息，多了不能嵌入,最后一组pixel不用于嵌入以防止错误
    assert(np.shape(secret_group)[0] <= np.shape(pixels_group)[0] - 1)
    #每一组secret_group计算得到一个d值，d为（2n+1）进制的一个数
    d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        #d代表一个（2n+1）进制的一个数
        d = 0
        for j in range(0,m):
            d += secret_group[i,j] * (2 ** (m - 1 - j))
            d_array[i] = d
    #-----------------------------------------------------------------------------------
    #开始进行嵌入
    embeded_pixels_group = pixels_group.copy()
    for i in range(0,num_secret_groups):
        d = d_array[i]
        fG = fG_array[i]
        j = int(d - fG) % moshu
        if (j > 0): #如果为0的话，则不进行修改
            if (j <= n) :
                embeded_pixels_group[i , j - 1]+=1
            else:
                embeded_pixels_group[i ,(2 * n + 1 - j) - 1]+=-1

    #-----------------------------------------------------------------------------------
    # 恢复，提取加密数据
    recover_d_array = np.zeros(num_secret_groups)
    for i in range(0,num_secret_groups):
        fG = 0
        for j in range(0,n):
            fG += (j + 1) * embeded_pixels_group[i,j]
        recover_d_array[i] = fG % moshu

    # 恢复出的和以前的应该是一致的
    assert(int((recover_d_array - d_array).sum()) == 0)
    #使用了多少pixel来进行嵌入
    num_pixels_changed = num_secret_groups * n
    #-----------------------------------------------------------------------------------
    #输出图像
    img_out = embeded_pixels_group.flatten()
    img_out = img_out[:512 * 512] #取前面的pixel
    #计算PSNR
    img_array_out = img_out.copy()
    #psnr = PSNR(image_array,img_array_out)
    imgpsnr1 = image_array[0:num_pixels_changed]
    imgpsnr2 = img_array_out[0:num_pixels_changed]
    psnr = PSNR(imgpsnr1, imgpsnr2)
    #print('EMD_2006 PSNR: %.2f' % psnr)
    print('EMD_2006 k=%d n=%d PSNR: %.2f' % (k,n,psnr))
    #csnr = CSNR(image_array,img_array_out)
    #print('EMD_2006 CSNR: %.2f' % csnr)
    #重组图像
    img_out = img_out.reshape(512,512)
    img_out = Image.fromarray(img_out)
    #img_out.show()
    img_out = img_out.convert('L')
    (filepath, tempfilename) = os.path.split(image_file_name)
    (originfilename, extension) = os.path.splitext(tempfilename)
    new_file = filepath + '\\Output1\\' + originfilename + "_EMD_k_" + str(k)+ "_n_"+str(n)+ ".png"
    print('new_file',new_file)
    img_out.save(new_file, 'png')

    return 0


#proof()

#需要嵌入的信息,用整形0,1两种数值，分别表示二进制的0,1
np.random.seed(1024)
s_data = np.random.randint(0,2,1000) #49000 #98000 72000, 262144, 524288, 786432, 1048576
path = r"D:\information hiding\EMD\512_512"
SaveResult('start')

for file in os.listdir(path):
    file_path = os.path.join(path, file)
    #if "Pepper.png" not in file_path:
    #    continue
    #if "Tiffany.png" not in file_path:
    #    continue
    if os.path.isfile(file_path):
        print(file_path)
        #开始仿真
        img = Image.open(file_path,"r")
        img = img.convert('L')
        #img.show()

        # 将二维数组，转换为一维数组
        img_array1 = np.array(img)
        img_array2 = img_array1.reshape(img_array1.shape[0] * img_array1.shape[1])
        #print(img_array2)
        # 将二维数组，转换为一维数组
        img_array3 = img_array1.flatten()
        #print(img_array3)

        #调用函数
        EMD_2006(img_array3,s_data,4,2,file_path)
        #EMD_2006(img_array3,s_data,4,4,file_path)
       #EMD_2006(img_array3, s_data,4,8,file_path)

        #KKWW_2016(img_array3,s_data,5,2,file_path)
        # JY09(img_array3,s_data,1,file_path)
        # JY09(img_array3,s_data,2,file_path)
        #
        # for k in range(2,5):
        #     SB19(img_array3,s_data,k,file_path)
        #SB19(img_array3,s_data,1,file_path)


SaveResult('end')
time.sleep(10)
