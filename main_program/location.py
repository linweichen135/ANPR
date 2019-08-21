###########################################################################################################################################################################
import cv2
import os
import numpy as np 
import PIL.Image as Image
import PIL.ImageEnhance as IME
def findPlateNumberRegion(img):
    height_ori = img.shape[0]
    region = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print("contours lenth is :%s" % (len(contours)))
    list_rate = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < 1250 or area>10000:
            continue
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = float(width) / float(height)
        rate = getxyRate(cnt)
        if ratio > 6 or ratio < 1:
            continue
        region.append(box)
        list_rate.append(ratio)         
    if list_rate!=[]:  #########################     change here 
        index,index2,index3,region_num = getSatifyestBox(list_rate, region, height_ori)
        return_value=region[index]
        if region_num>1:
            return_value2=region[index2]
            if region_num>2:
                return_value3=region[index3]
            else:  
                return_value3=[]
        else:
            return_value2=[]
            return_value3=[]                    
    else:
        return_value=[]
        return_value2=[]
        return_value3=[]
        region_num=0        
    return return_value,return_value2,return_value3,region_num
def getSatifyestBox(list_rate, region, height_ori):#########################     change here 
    count=0
    region_list=[]
    for index, key in enumerate(region):        
        list_rate[index] = abs(list_rate[index] - 2.2)
        if list_rate[index]<1.5:
            count+=1
            region_list.append(max(key[0][1],key[1][1],key[2][1],key[3][1]))
        else:
            region_list.append(0)               
    #print(list_rate)
    #print(region_list)
    if count>1:
        index = region_list.index(max(region_list))
        region_list[index]=-1
        index2=region_list.index(max(region_list))
        region_list[index2]=-1
        if len(region_list)>2:
            index3=region_list.index(max(region_list))
            region_num=3
        else:
            index3=100
            region_num=2    
    else:
        index = list_rate.index(min(list_rate))
        index2 = 100
        index3= 100
        region_num=1
    #print(index)
    return index,index2,index3,region_num
def getxyRate(cnt):
    x_height = 0
    y_height = 0
    x_list = []
    y_list = []
    for location_value in cnt:
        location = location_value[0]
        x_list.append(location[0])
        y_list.append(location[1])
    x_height = max(x_list) - min(x_list)
    y_height = max(y_list) - min(y_list)
    return x_height * (1.0) / y_height * (1.0)
def gamma_trans(img,gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)    
def location(img,blur):
    #img = Image.open(file)#.convert('L')
    img = img.resize((500,281),Image.ANTIALIAS)
    #file = "adjusted_data/"+str(num)+".jpg"
    #img.save(file)
    #if opt.crop==1:
    img=img.crop((img.size[0]*0.2,img.size[1]*0.2,img.size[0]*0.8,img.size[1]*0.8))
    img=np.asarray(img)
    img=img[:,:,::-1]
    original=img
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img3=img
    img=cv2.equalizeHist(img)
    img=gamma_trans(img,1.5)
    img2=gamma_trans(img3,0.7)
    #img=cv2.equalizeHist(img)
    #img=gamma_trans(img,2)
    #cv2.imwrite(file,img)        
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray=img
    gaussian = cv2.GaussianBlur(img3, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize = 3)
    ret, binary1 = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    gaussian = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize = 3)
    ret, binary2 = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    gaussian = cv2.GaussianBlur(img2, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    median = cv2.medianBlur(gaussian, 5)
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize = 3)
    ret, binary3 = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    binary=cv2.bitwise_or(binary1,binary2)
    binary=cv2.bitwise_or(binary,binary3)    
    #cv2.imwrite('croped_0228/'+str(num)+'_binary.jpg',binary)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    dilation = cv2.dilate(binary, element2, iterations = 3)#########################     change here 
    erosion = cv2.erode(dilation, element1, iterations = 5)#########################     change here 
    dilation2 = cv2.dilate(erosion, element2,iterations = 3)#########################     change here 
    #print (type(dilation2))    
    #output_file='croped_0228/'+str(num)+'_mask1.jpg'
    #cv2.imwrite(output_file,dilation2)      
    region,region2,region3,region_num = findPlateNumberRegion(dilation2)    
    out,ori,success,slice_num,images,image_bw=test_region(region,original)
    if success==False and region_num>1:
        out,ori,success,slice_num,images,image_bw=test_region(region2,original)
    if success==False and region_num>2:
        out,ori,success,slice_num,images,image_bw=test_region(region3,original)
    binary = Image.fromarray(binary)
    dilation2 = Image.fromarray(dilation2)
    ori = Image.fromarray(ori)
    if success==True:
        image_bw = Image.fromarray(image_bw)        
    else:
        image_bw=[]
    if(blur):
        count=0
        for i in images:
            image_temp=cv2.cvtColor(np.asarray(images[count]),cv2.COLOR_RGB2BGR)
            image_temp=cv2.blur(image_temp,(5,5))
            _,image_temp=cv2.threshold(image_temp,127,255,cv2.THRESH_BINARY)
            images[count]=Image.fromarray(cv2.cvtColor(image_temp,cv2.COLOR_BGR2RGB))
            count+=1    
    return out,binary,dilation2,ori,success,slice_num,images,image_bw    
def test_region(region,original):
    images = []    
    if region != []:
        if region.max(0)[1]>original.shape[0]:
            top = original.shape[0]
        else:
            top = region.max(0)[1]
        if region.min(0)[1]<0 :
            bottom =0
        else:
            bottom =region.min(0)[1] 
        if region.max(0)[0]>original.shape[1]:
            right = original.shape[1]
        else:
            right = region.max(0)[0]
        if region.min(0)[0]<0 :
            left =0
        else:
            left =region.min(0)[0] 
        #print(top,bottom,left,right)
        out=original[bottom:top,left:right]        
        #cv2.imwrite('croped_0228/'+str(num)+'_'+str(num2)+'_original.jpg',out)
        slice_num,images,image_bw=spilt_recog(out)
        #print(slice_num)
        if slice_num<6:
            x_len=abs(right-left)
            y_len=abs(top-bottom)  
            out2=original[int(max((bottom-y_len*0.1),0)):int(min((top+y_len*0.1),original.shape[0])),int(max((left-x_len*0.25),0)):int(min((right+x_len*0.25),original.shape[1]))]
            slice_num,images,image_bw=spilt_recog(out2)
            #print(slice_num)
            if slice_num<6:                       
                out=[]
                list=[]
                #cv2.imwrite('croped_0228/'+str(num)+'_'+str(num)+'crop_2_fail.jpg',out2)
                success=False
                return out,original,success,slice_num,images,image_bw   
            else:
                #output_file='croped_0228/'+str(num)+'_correct.jpg'
                #cv2.imwrite(output_file,out2)
                #croped_2+=1
                #print("saved2")
                success=True
                out2=Image.fromarray(cv2.cvtColor(out2,cv2.COLOR_BGR2RGB))
                original=original.copy()
                original= cv2.rectangle(original, (int(min((right+x_len*0.25),original.shape[1])),int(min((top+y_len*0.1),original.shape[0]))),(int(max((left-x_len*0.25),0)),int(max((bottom-y_len*0.1),0))) ,(0, 0, 255),2)
                return out2,original,success,slice_num,images,image_bw      
        else:        
            #output_file='croped_0228/'+str(num)+'_correct.jpg'
            #cv2.imwrite(output_file,out)
            #croped_1+=1
            #print("saved1")
            success=True
            out=Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
            original=original.copy()
            original= cv2.rectangle(original, (right,top),(left,bottom) , (0, 0, 255), 2)
            return out,original,success,slice_num,images,image_bw
    else:                
        out=[]
        slice_num=0    
        #print('no result!')
        #no_result+=1
        success=False
        list=[]
        image_bw=[]         
        return out,original,success,slice_num,images,image_bw
def spilt_recog(out2):
    
    plate_list=[]
    how_many_words = 0
    #out2=cv2.cvtColor(out2, cv2.COLOR_GRAY2BGR)
    #Algorithm
    out2_mark=out2
    out2_ori=out2
    out2=cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    r,out3=cv2.threshold(out2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    out_result = cv2.cvtColor(out3,cv2.COLOR_GRAY2BGR)
    r,out3_b=cv2.threshold(out2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    element1=cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    #element2=cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    out3=cv2.dilate(out3,element1,iterations=1)
    #out3_b=cv2.dilate(out3_b,element2,iterations=1)
    element2=cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    out3_b=cv2.erode(out3_b,element2,iterations=1)                
    #white_area,white_region=find_white_area(out3)
    out4=cv2.bitwise_and(out3,out3_b)

    
    #file = "cut_test/"+str(num)+"_out3.jpg"
    #cv2.imwrite(file,out3)
    #file = "cut_test/"+str(num)+"_out3_b.jpg"  
    #cv2.imwrite(file,out3_b)
    
    height_ori=out2_ori.shape[0]
    width_ori=out2_ori.shape[1]
    area_ori=height_ori*width_ori
    #print("original area is: %s" % area_ori)
    
    num_legal_cnt = 0
    region = []
    area_rect_list = []
    box_list = []
    height_rect_list = []
    width_rect_list = []
    valid_rect_list = []
    exception_list = []
    remove_list=[]
    contours, hierarchy = cv2.findContours(out4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print("contours lenth is :%s" % (len(contours)))
    
    #Find valid region
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area == 0:
            continue
        #area_ratio = int(area_ori/area)
        #if((area_ratio>50) or (area_ratio<9)):
        #    continue
        #if area < 200 or area > 220:
        #    continue
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        height_rect=(max(box[0][1],box[1][1],box[2][1],box[3][1])-min(box[0][1],box[1][1],box[2][1],box[3][1]))
        width_rect =(max(box[0][0],box[1][0],box[2][0],box[3][0])-min(box[0][0],box[1][0],box[2][0],box[3][0])) 
        area_rect = width_rect*height_rect
        if area_rect == 0 :
            continue
        area_ratio = (area_ori/area_rect)
        if (area_ratio < 5) or (area_rect < 20) or (height_rect < 10):
            continue
        area_rect_list.append(area_rect)
        box_list.append(box)
        #print(box)
        height_rect_list.append(height_rect)
        width_rect_list.append(width_rect)
        num_legal_cnt+=1
        #print(area_rect)
        #area_ratio = (area_ori/area_rect)
        #if((area_ratio>40) or (area_ratio<10)):
        #    continue
    if(num_legal_cnt == 0):
        return 0, plate_list, out4
    
    #Sort with height
    for i in range (num_legal_cnt-1):
        for j in range (num_legal_cnt-1-i):
            if height_rect_list[j] > height_rect_list[j+1]:
                temp = height_rect_list[j]
                height_rect_list[j] = height_rect_list[j+1]
                height_rect_list[j+1] = temp
                temp = width_rect_list[j]
                width_rect_list[j] = width_rect_list[j+1]
                width_rect_list[j+1] = temp
                temp = area_rect_list[j]
                area_rect_list[j] = area_rect_list[j+1]
                area_rect_list[j+1] = temp
                temp_box = box_list[j]
                box_list[j] = box_list[j+1]
                box_list[j+1] = temp_box
    #print(height_rect_list)
    #print(num_legal_cnt)
    
    #Find target with standard height
    arg = (num_legal_cnt % 2)
    #print(arg)
    mid_num_list = ((int(num_legal_cnt/2) + 1) if arg else int(num_legal_cnt/2))
    if(num_legal_cnt == 1):
        mid_num_list = 0 
    #print(mid_num_list)
    std_height = height_rect_list[mid_num_list]
    std_bot=max(box_list[mid_num_list][0][1],box_list[mid_num_list][1][1],box_list[mid_num_list][2][1],box_list[mid_num_list][3][1])
    if std_bot > height_ori-1:
        std_bot = height_ori-1
    std_top=min(box_list[mid_num_list][0][1],box_list[mid_num_list][1][1],box_list[mid_num_list][2][1],box_list[mid_num_list][3][1])
    if std_top < 0:
        std_top = 0
    std_mid_point=int((std_bot+std_top)/2)
    #print("std: "+str(std_mid_point))
    for i in range (num_legal_cnt):
        test_bot=max(box_list[i][0][1],box_list[i][1][1],box_list[i][2][1],box_list[i][3][1])
        if test_bot > height_ori-1:
            test_bot = height_ori-1
        test_top=min(box_list[i][0][1],box_list[i][1][1],box_list[i][2][1],box_list[i][3][1])
        if test_top < 0:
            test_top = 0
        test_mid_point = int((test_bot+test_top)/2)
        #print(test_bot)
        #print(test_top)
        #print(test_mid_point)
        if (height_rect_list[i] > (std_height-5))  and (height_rect_list[i] < (std_height+5)) and (test_mid_point > (std_mid_point-10)) and (test_mid_point < (std_mid_point+10)):
            valid_rect_list.append(box_list[i])
            remove_list.append(i)
            #print(test_mid_point)
                   
    how_many_words=len(valid_rect_list)    
    #print(how_many_words)
    if(how_many_words == 0):
        return 0, plate_list, out4
    
    #Exception
        #Sort with left side
    avg_exp = 0    
    if((how_many_words)<7):
        temp_exp = 0
        for i in range (how_many_words):
            bot=max(valid_rect_list[i][0][1],valid_rect_list[i][1][1],valid_rect_list[i][2][1],valid_rect_list[i][3][1])
            if bot > height_ori-1:
                bot = height_ori-1
            top=min(valid_rect_list[i][0][1],valid_rect_list[i][1][1],valid_rect_list[i][2][1],valid_rect_list[i][3][1])
            if top < 0:
                top = 0
            temp_exp += int((bot + top)/2)
        avg_exp = int((temp_exp)/(how_many_words))
    #while((how_many_words)<6):
        for i in range (num_legal_cnt):
            temp_exp = 0
            if i not in remove_list:
                bot=max(box_list[i][0][1],box_list[i][1][1],box_list[i][2][1],box_list[i][3][1])
                if bot > height_ori-1:
                    bot = height_ori-1
                top=min(box_list[i][0][1],box_list[i][1][1],box_list[i][2][1],box_list[i][3][1])
                if top < 0:
                    top = 0
                temp_exp = int((bot + top)/2)
                height_exp = int(bot-top)
                ratio_exp = height_exp/height_ori
                if (temp_exp>(avg_exp-13)) and (temp_exp<(avg_exp+13)) and (ratio_exp < 0.78) and (height_exp > (std_height-3))  :
                    valid_rect_list.append(box_list[i])
                    remove_list.append(i)
                    how_many_words +=1
    if (how_many_words<4) or (how_many_words > 9):
        return 0, plate_list, out4  
    
    #Sort with left side
    for i in range (how_many_words-1):
        for j in range (how_many_words-1-i):
            if min(valid_rect_list[j][0][0],valid_rect_list[j][1][0],valid_rect_list[j][2][0],valid_rect_list[j][3][0]) > min(valid_rect_list[j+1][0][0],valid_rect_list[j+1][1][0],valid_rect_list[j+1][2][0],valid_rect_list[j+1][3][0]):
                temp = valid_rect_list[j]
                valid_rect_list[j] = valid_rect_list[j+1]
                valid_rect_list[j+1] = temp
    
    #Get target region
    for i in range (how_many_words):
        valid_rect_list[i].astype(int)
        right=max(valid_rect_list[i][0][0],valid_rect_list[i][1][0],valid_rect_list[i][2][0],valid_rect_list[i][3][0])
        if right > width_ori-1:
            right = width_ori-1
        left=min(valid_rect_list[i][0][0],valid_rect_list[i][1][0],valid_rect_list[i][2][0],valid_rect_list[i][3][0])
        if left < 0:
            left = 0
        bot=max(valid_rect_list[i][0][1],valid_rect_list[i][1][1],valid_rect_list[i][2][1],valid_rect_list[i][3][1])
        if bot > height_ori-1:
            bot = height_ori-1
        top=min(valid_rect_list[i][0][1],valid_rect_list[i][1][1],valid_rect_list[i][2][1],valid_rect_list[i][3][1])
        if top < 0:
            top = 0
        #print("left is %s" % left)
        #print("right is %s" % right)
        #print("bot is %s" % bot)
        #print("top is %s" % top)
        cut_plate=out_result[top:bot,left:right]
        #file = "cut_test/"+str(num)+"_how_"+str(i)+".jpg"
        #cv2.imwrite(file,cut_plate)
        cut_plate = Image.fromarray(cv2.cvtColor(cut_plate,cv2.COLOR_BGR2RGB))
        cut_height=cut_plate.height
        cut_width=cut_plate.width
        #cut_back = np.ones((62,62,3),np.uint8)
        #cv2.imshow(cut_plate)
        cut_back = Image.new("RGB",(62,62),(255,255,255))
        #cut_back[:,:] = (255,255,255)
        if cut_width < cut_height:
            new_cut_width = int(cut_width * (62.0 / cut_height))
            #print(new_cut_width)
            cut_plate_result = cut_plate.resize((new_cut_width,62))
            #rint(cut_plate_result)
            offset =int((62 - new_cut_width) / 2)
            cut_back.paste(cut_plate_result,(offset,0))
            #cut_back[:,offset:offset+new_cut_width] = cut_plate_result
        else:
            new_cut_height = int(cut_height * (62.0 / cut_width))
            cut_plate_result = cut_plate.resize((62,new_cut_height))
            offset =int((62 - new_cut_height) / 2)
            cut_back.paste(cut_plate_result,(offset,0))
            #cut_back[offset:offset+new_cut_height,:] = cut_plate_result
        #print(cut_plate_result)
        plate_list.append(cut_back)
        #cut_back = cv2.cvtColor(np.asarray(cut_back),cv2.COLOR_RGB2BGR)
        #file = "cut_test/"+str(num)+"_result_"+str(i)+".jpg"
        #cv2.imwrite(file,cut_back)
    
    #Draw rectangular for debuging               
    #for i in range (num_legal_cnt):
    #    box = box_list[i]    
    #   out2_mark = cv2.drawContours(out2_mark, [box.astype(int)], -1, (0, 0, 255), 1)
                
    #print("number of legal contours is: %s" % (num_legal_cnt))
                   
    #file = "cut_test/"+str(num)+"_result.jpg"    
    #cv2.imwrite(file,out2_mark) 
    #file = "cut_test/"+str(num)+"_out.jpg"    
    #cv2.imwrite(file,out4)
    return how_many_words,plate_list,out4          

