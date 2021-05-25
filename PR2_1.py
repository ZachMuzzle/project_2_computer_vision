import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def matching(imgL, imgR, template, max_d, method,harris):  # template = kernel

    if (method == 'SAD' and harris != True):
        h, w = imgL.shape
        h_r,w_r = imgR.shape
        disparity = np.zeros((h, w), np.uint8)
        disparity2 = np.zeros((h_r,w_r),np.uint8)
        disparity.shape = h, w
        disparity2.shape = h_r,w_r
        template_half = int(template / 2)
        # max_d_off = 255 / max_d

        ##########################################
        for i in tqdm(range(template_half, h - template_half), desc="Loading..."):
            for x in range(template_half, w - template_half):
                best_d = 0
                best_d2 = 0
                prev_sad = 65534
                prev_sad2 = 65534
                for d in range(0, max_d):
                    sad = 0
                    sad2 = 0
                    sad_temp = 0
                    sad_temp2 = 0
                    for v in range(-template_half, template_half):
                        for u in range(-template_half, template_half):
                            # left to right
                            sad_temp = np.abs(int(imgL[i + v, x + u]) - int(imgR[v + i, (x + u) - d]))
                            #right to left
                            sad_temp2 = np.abs(int(imgR[i + v, x + u]) - int(imgL[v + i, (x + u) - d]))
                            sad += sad_temp
                            sad2 += sad_temp2

                    if sad < prev_sad:
                        prev_sad = sad
                        best_d = d

                    if sad2 < prev_sad2:
                        prev_sad2 = sad2
                        best_d2 = d

                disparity[i, x] = best_d * (255 / max_d)
                disparity2[i, x] = best_d2 * (255 / max_d)
        ################################################################
        # disparity_sad = np.argmin(disparity,axis=0 )
        # disparity_sad = np.uint8(disparity* 255/max_d)
        # disparity_sad = disparity * 255/max_d
        #disparity_sad = cv2.equalizeHist(disparity)
        #disparity_sad2 = cv2.equalizeHist(disparity2)
        return disparity,disparity2

    if (method == 'SAD' and harris == True):
        imgL_harris = cv2.cornerHarris(imgL, 2, 3, 0.04)
        imgR_harris = cv2.cornerHarris(imgR, 2, 3, 0.04)

        conner_threshold_l = imgL_harris.max() * 0.01
        conner_threshold_r = imgR_harris.max() * 0.01
        imgL_harris_t = (imgL_harris > conner_threshold_l) *1
        imgR_harris_t = (imgR_harris > conner_threshold_r) * 1
        #plt.imshow(imgL_harris_t,cmap='gray')
        #plt.show()
        h, w = imgL_harris_t.shape
        disparity = np.zeros((h, w), np.uint8)
        disparity.shape = h, w
        template_half = int(template / 2)
        # max_d_off = 255 / max_d

        ##########################################
        for i in tqdm(range(template_half, h - template_half), desc="Loading..."):
            for x in range(template_half, w - template_half):
                best_d = 0
                prev_sad = 65534

                for d in range(0, max_d):
                    sad = 0
                    sad_temp = 0
                    for v in range(-template_half, template_half):
                        for u in range(-template_half, template_half):
                            sad_temp = np.abs(int(imgL_harris_t[i + v, x + u]) - int(imgR_harris_t[v + i, (x + u) - d]))
                            sad += sad_temp

                    if sad < prev_sad:
                        prev_sad = sad
                        best_d = d

                disparity[i, x] = best_d * (255 / max_d)

        #disparity_sad_harris = cv2.equalizeHist(disparity)
        return disparity
        # disparity_sad = np.argmin(disparity, axis=2)
        # disparity_sad = np.uint8(disparity_sad * 255 / max_d)
        # disparity_sad = cv2.equalizeHist(disparity_sad)
        # return disparity_sad

    if (method == 'SSD' and harris != True):

        h, w = imgL.shape
        h_r, w_r = imgR.shape
        disparity = np.zeros((h, w), np.uint8)
        disparity2  = np.zeros((h_r,w_r),np.uint8)
        disparity.shape = h, w
        disparity2.shape = h_r, w_r
        template_half = int(template / 2)
        # max_d_off = 255 / max_d

        ##########################################
        for i in tqdm(range(template_half, h - template_half), desc="Loading..."):
            for x in range(template_half, w - template_half):
                best_d = 0
                best_d2 = 0
                prev_ssd = 70592
                prev_ssd2 = 70592

                for d in range(0, max_d):
                    ssd = 0
                    ssd2 = 0
                    ssd_temp = 0
                    ssd_temp2 = 0
                    for v in range(-template_half, template_half):
                        for u in range(-template_half, template_half):
                            ssd_temp = (int(imgL[i + v, x + u]) - int(imgR[v + i, (x + u) - d]))**2
                            ssd_temp2 = (int(imgR[i + v, x + u]) - int(imgL[v + i, (x + u) - d]))**2

                            ssd += ssd_temp
                            ssd2 += ssd_temp2
                    ## store ssd into prev_ssd if less than prev_ssd
                    if ssd < prev_ssd:
                        prev_ssd = ssd
                        best_d = d

                    if ssd2 < prev_ssd2:
                        prev_ssd2 = ssd2
                        best_d2 = d

                disparity[i, x] = best_d * (255 / max_d)
                disparity2[i, x] = best_d2 * (255 / max_d)

        ################################################################
        # disparity_sad = np.argmin(disparity,axis=0 )
        # disparity_sad = np.uint8(disparity* 255/max_d)
        # disparity_sad = disparity * 255/max_d
        # disparity_ssd = cv2.equalizeHist(disparity)
        return disparity,disparity2

    if (method == 'SSD' and harris == True):
        imgL_harris = cv2.cornerHarris(imgL, 2, 3, 0.04)
        imgR_harris = cv2.cornerHarris(imgR, 2, 3, 0.04)

        conner_threshold_l = imgL_harris.max() * 0.01
        conner_threshold_r = imgR_harris.max() * 0.01
        imgL_harris_t = (imgL_harris > conner_threshold_l) * 1
        imgR_harris_t = (imgR_harris > conner_threshold_r) * 1
        # plt.imshow(imgL_harris_t,cmap='gray')
        # plt.show()
        h, w = imgL_harris_t.shape
        disparity = np.zeros((h, w), np.uint8)
        disparity.shape = h, w
        template_half = int(template / 2)
        # max_d_off = 255 / max_d

        ##########################################
        for i in tqdm(range(template_half, h - template_half), desc="Loading..."):
            for x in range(template_half, w - template_half):
                best_d = 0
                prev_ssd = 70592

                for d in range(0, max_d):
                    ssd = 0
                    ssd_temp = 0
                    for v in range(-template_half, template_half):
                        for u in range(-template_half, template_half):
                            ssd_temp = (int(imgL_harris_t[i + v, x + u]) - int(imgR_harris_t[v + i, (x + u) - d]))**2
                            ssd += ssd_temp

                    if ssd < prev_ssd:
                        prev_ssd = ssd
                        best_d = d

                disparity[i, x] = best_d * (255 / max_d)

        #disparity_ssd_harris = cv2.equalizeHist(disparity)
        return disparity

    if (method == 'NCC' and harris != True):
        h, w = imgL.shape
        h_r, w_r = imgR.shape
        disparity = np.zeros((h, w), np.uint8)
        disparity.shape = h, w
        template_half = int(template / 2)
        # max_d_off = 255 / max_d

        ##########################################
        for i in tqdm(range(template_half, h - template_half), desc="Loading..."):
            for x in range(template_half, w - template_half):
                best_d = 0
                prev_ncc = 65534

                l_mean = 0
                r_mean = 0
                n = 0

                for d in range(0, max_d):
                    ncc = 0
                    ncc_temp = 0
                    for v in range(-template_half, template_half + 1):
                        for u in range(-template_half, template_half + 1):
                            # ssd_temp = int(imgL[i + v, x + u]) - int(imgR[v + i, (x + u) - d])
                            # ssd += ssd_temp * ssd_temp
                            l_mean += int(imgL[i + v, x + u])
                            # print(l_mean)
                            r_mean += int(imgR[i + v, (x + u) - d])
                            n += 1

                    l_mean = l_mean / n
                    # print(l_mean)
                    r_mean = r_mean / n
                    # print(r_mean)

                    l_r = 0
                    l_var = 0
                    r_var = 0

                    for v in range(-template_half, template_half + 1):
                        for u in range(-template_half, template_half + 1):
                            l = imgL[i + v, x + u] - l_mean
                            r = imgR[i + v, (x + u) - d] - r_mean
                            # print(r)

                            l_r += l * r
                            # print(l_r)
                            l_var += l ** 2
                            # print(l_var)
                            r_var += r ** 2
                            # print(r_var)
                            ncc_temp = -l_r / np.sqrt(l_var * r_var)
                            ncc += ncc_temp

                    if ncc < prev_ncc:
                        prev_ncc = ncc
                        best_d = d

                    disparity[i, x] = best_d * 255 / (max_d)

        ################################################################
        # disparity_sad = np.argmin(disparity,axis=0 )
        # disparity_sad = np.uint8(disparity* 255/max_d)
        # disparity_sad = disparity * 255/max_d
        # disparity_ssd = cv2.equalizeHist(disparity)
        return disparity
    if (method == 'NCC' and harris == True):
        imgL_harris = cv2.cornerHarris(imgL, 2, 3, 0.04)
        imgR_harris = cv2.cornerHarris(imgR, 2, 3, 0.04)

        conner_threshold_l = imgL_harris.max() * 0.01
        conner_threshold_r = imgR_harris.max() * 0.01
        imgL_harris_t = (imgL_harris > conner_threshold_l) * 1
        imgR_harris_t = (imgR_harris > conner_threshold_r) * 1

        h, w = imgL.shape
        disparity = np.zeros((h, w), np.uint8)
        disparity.shape = h, w
        template_half = int(template / 2)
        # max_d_off = 255 / max_d

        ##########################################
        for i in tqdm(range(template_half, h - template_half), desc="Loading..."):
            for x in range(template_half, w - template_half):
                best_d = 0
                prev_ncc = 65534

                l_mean = 0
                r_mean = 0
                n = 0

                for d in range(0, max_d):
                    ncc = 0
                    ncc_temp = 0
                    for v in range(-template_half, template_half + 1):
                        for u in range(-template_half, template_half + 1):
                            # ssd_temp = int(imgL[i + v, x + u]) - int(imgR[v + i, (x + u) - d])
                            # ssd += ssd_temp * ssd_temp
                            l_mean += int(imgL_harris_t[i + v, x + u])
                            # print(l_mean)
                            r_mean += int(imgR_harris_t[i + v, (x + u) - d])
                            n += 1

                    l_mean = l_mean / n
                    # print(l_mean)
                    r_mean = r_mean / n
                    # print(r_mean)

                    l_r = 0
                    l_var = 0
                    r_var = 0

                    for v in range(-template_half, template_half + 1):
                        for u in range(-template_half, template_half + 1):
                            l = imgL_harris_t[i + v, x + u] - l_mean
                            r = imgR_harris_t[i + v, (x + u) - d] - r_mean
                            # print(r)

                            l_r += l * r
                            # print(l_r)
                            l_var += l ** 2
                            # print(l_var)
                            r_var += r ** 2
                            # print(r_var)
                            ncc_temp = -l_r / np.sqrt(l_var * r_var)
                            ncc += ncc_temp

                    if ncc < prev_ncc:
                        prev_ncc = ncc
                        best_d = d

                    disparity[i, x] = best_d * 255 / (max_d)

        ################################################################
        # disparity_sad = np.argmin(disparity,axis=0 )
        # disparity_sad = np.uint8(disparity* 255/max_d)
        # disparity_sad = disparity * 255/max_d
        # disparity_ssd = cv2.equalizeHist(disparity)
        return disparity

#fill gaps that are zeros
def averaging(img, window):
    kernel = np.ones((window,window),np.float32) / 25
    img = cv2.filter2D(img,-1,kernel)

    return img
#check if pixels are same
def validity(left, right):
    h,w = left.shape
    h1,w1 = right.shape
    #disparity = np.zeros()
    for i in range(0,h,1):
        for j in range(0,w,1):
            if left[i,j] != right[i,j]:
                left[i,j] = 0

    for i in range(0,h1,1):
        for j in range(0,w1,1):
            if right[i,j] != left[i,j]:
                right[i,j] = 0

    return left,right
# Start of main
left = cv2.imread('left_img.jpg', 0)
right = cv2.imread('right_img.jpg', 0)






stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(left, right)
disparity2 = stereo.compute(right,left)

print(disparity)
fig = plt.figure(figsize=(10,10))
plt.subplot(131);plt.imshow(left,cmap='gray');plt.title('Left img')
plt.subplot(132);plt.imshow(right,cmap='gray');plt.title('right img')
plt.subplot(133);plt.imshow(disparity,cmap='gray');plt.title('disparity img')
#plt.imshow(disparity, 'gray')
plt.show()
fig = plt.figure(figsize=(10,10))
plt.subplot(131);plt.imshow(right,cmap='gray');plt.title('Left img')
plt.subplot(132);plt.imshow(left,cmap='gray');plt.title('right img')
plt.subplot(133);plt.imshow(disparity2,cmap='gray');plt.title('disparity img')
plt.show()
print("END")

# call our matching function for SAD
dis_SAD,dis_SAD2 = matching(left,right,10, max_d=10, method='SAD',harris=False)
plt.imshow(dis_SAD,cmap='gray')
plt.show()
#plt.imshow(dis_SAD2,cmap='gray')
#plt.show()

# call matching for SSD
dis_SSD,dis_SSD2 = matching(left,right,10, max_d=10, method='SSD',harris=False)
plt.imshow(dis_SSD,cmap='gray')
plt.show()


# call matching for NCC
# I didn't do right to left with ncc as ncc takes a long time to complete
dis_NCC = matching(left,right,10,max_d=10,method='NCC',harris=False)
plt.imshow(dis_NCC,cmap='gray')
plt.show()

# harris corner detector

imgL_harris = cv2.cornerHarris(left, 2, 3, 0.04)
imgR_harris = cv2.cornerHarris(right, 2, 3, 0.04)
fig = plt.figure(figsize=(10, 10))

plt.subplot(121);
plt.imshow(imgL_harris > 0.01 * imgL_harris.max(), cmap='gray');
plt.title("Left Harris")
plt.subplot(122);
plt.imshow(imgR_harris > 0.01 * imgR_harris.max(), cmap='gray');
plt.title("Right Harris")
plt.show()

fig = plt.figure(figsize=(10, 10))
conner_threshold = imgL_harris.max() * 0.01
t_l = (imgL_harris > conner_threshold) * 1
conner_threshold_r = imgR_harris.max() * 0.01
t_r = (imgR_harris > conner_threshold_r) * 1
plt.subplot(121);
plt.imshow(t_l, cmap='gray');
plt.title("Left Harris Threshold")
plt.subplot(122);
plt.imshow(t_r, cmap='gray');
plt.title("Right Harris threshold")
plt.show()

#harris for sad
disp_harris_t_sad = matching(left, right, 6, max_d=10, method='SAD', harris=True)
plt.imshow(disp_harris_t_sad, cmap='gray')
plt.show()

plt.imshow(averaging(disp_harris_t_sad,5),cmap='gray')
plt.show()

# harris for ssd
disp_harris_t_ssd = matching(left, right, 6, max_d=10, method='SSD', harris=True)
plt.imshow(disp_harris_t_ssd, cmap='gray')
plt.show()

plt.imshow(averaging(disp_harris_t_ssd,5),cmap='gray')
plt.show()

# harris for ncc

disp_harris_t_ncc = matching(left, right, 6, max_d=10, method='NCC', harris=True)
plt.imshow(disp_harris_t_ncc, cmap='gray')
plt.show()

plt.imshow(averaging(disp_harris_t_ncc,5),cmap='gray')
plt.show()


#validate part

validate_d,validate_d2 = validity(dis_SSD,dis_SSD2)

plt.imshow(validate_d,cmap='gray')
plt.show()
plt.imshow(validate_d2,cmap='gray')
plt.show()

#Average validate
plt.imshow(averaging(validate_d,6),cmap='gray')
plt.show()
plt.imshow(averaging(validate_d2,6),cmap='gray')
plt.show()
