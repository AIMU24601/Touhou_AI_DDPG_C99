import cv2
import numpy as np
from PIL import ImageGrab
import matplotlib.pyplot as plt
import time

""""
何かごちゃごちゃ書いてありますが要するに走らせるとimage2.jpgって名前でゲームのスクリーンショットが保存されます。ここで確認してから他のプログラムを走らせる。
"""

ORI = "image.jpg"
DEATHCHECK_FILE = "Deathcheck_color.jpg"
COMPARING_FILE_1 = "Notdeathcheck.jpg"
COMPARING_FILE_2 = "Pause.jpg"
IMG = "image1.jpg"
SCORE = "Score.png"
SCORE_1 = "Score_1.pmg"
SCORE_2 = "Score_2.png"
SCORE_3 = "Score_3.png"
SCORE_4 = "Score_4.png"
tt = "tt.png"
img = cv2.imread(ORI)
img1 = cv2.imread(DEATHCHECK_FILE)
img2 = cv2.imread(COMPARING_FILE_1)
img3 = cv2.imread(COMPARING_FILE_2)
img99 = cv2.imread("image99.jpg")
img999 = cv2.imread("image999.jpg")
img9999 = cv2.imread("image9999.jpg")
img99999 = cv2.imread("image99999.jpg")
img999999 = cv2.imread("image999999.jpg")
img9999999 = cv2.imread("image9999999.jpg")
img_sent = cv2.imread(IMG)
img_P = cv2.imread(SCORE)
img_P_4 = cv2.imread(SCORE_4) # 0.4
#img_P_4 = cv2.imread(SCORE_4, cv2.IMREAD_GRAYSCALE)
img_T = cv2.imread(tt)
img_T_2 = cv2.imread("ttt.png") #0.7
img_c = cv2.imread("chapter.png") #0.8
img_d = cv2.imread("Death.png")
img_enter = img9999999
img_temp = img_d
threshold = 0.8
#cv2.imshow("image", img1)
#cv2.waitKey(0)
#print(img1)

"""
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
detector = cv2.AKAZE_create()
kp1, des1 = detector.detectAndCompute(img1, None)
print(kp1, des1)

print("TARGET_FILE: %s" % (TARGET_FILE))

kp2, des2 = detector.detectAndCompute(img2, None)
matches = bf.match(des1, des2)
print(des1, des2)
dist = [m.distance for m in matches]
print(dist)
print("Notdeathcheck: %s" % (COMPARING_FILE_1))
#ret = sum(dist) / len(dist)
#print("Notdeathcheck: %i" % (ret))

def cos_sim(v1, v2):
    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.nnorm(v2))

print(cos_sim(img1, img2))
"""

"""
target_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
#print(target_hist)
comparing_hist_1 = cv2.calcHist([img2], [0], None, [256], [0, 256])
ret = cv2.compareHist(target_hist, comparing_hist_1, 0)
print(COMPARING_FILE_1, ret)

comparing_hist_2 = cv2.calcHist([img3], [0], None, [256], [0, 256])
ret = cv2.compareHist(target_hist, comparing_hist_2, 0)
print(COMPARING_FILE_2, ret)

comparing_hist_3 = cv2.calcHist([img_g], [0], None, [256], [0, 256])
ret = cv2.compareHist(target_hist, comparing_hist_3, 0)
print(IMG, ret)
"""

"""
target_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
comparing_hist = cv2.calcHist([img2], [0], None, [256], [0, 256])
ret = cv2.compareHist(target_hist, comparing_hist, 0)
print(ret)
"""

time.sleep(2)
x = 284
y = 290
w = 670
h = 740
img_g = ImageGrab.grab((x, y, w, h))
img_g = np.asarray(img_g, dtype="uint8")
img_g = cv2.cvtColor(img_g, cv2.COLOR_RGB2BGR)
#img_g = cv2.cvtColor(img_g, cv2.COLOR_RGB2GRAY)
cv2.imwrite("image2.jpg", img_g)
print("saved")

#match_result = cv2.matchTemplate(img, img_sent, cv2.TM_CCOEFF_NORMED)
match_result = cv2.matchTemplate(img_enter, img_temp, cv2.TM_CCOEFF_NORMED)

"""
print(np.shape(img))
print(np.shape(img_g))
#print(match_result)
threshold = 0.9
loc = np.where(match_result >- threshold)
print(loc)
print(np.shape(loc))

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
top_left = max_loc
print(img_g.shape[::-1])
w, h = img_g.shape[::-1]
bottom_right = (top_left[0] + w, top_left[1] + h)

result = cv2.imread(ORI)
cv2.rectangle(result, top_left, bottom_right, (255, 0, 0), 10)
cv2.imwrite("result.png", result)
"""

H, W, C = img.shape
h, w, c = img_P_4.shape
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
print(min_val, max_val, min_loc, max_loc)

locs = np.column_stack(np.where(match_result>=threshold))
print(np.where(match_result>=threshold))
print(locs)

fig, ax = plt.subplots(facecolor="w")
for y, x in locs:
    ax.add_patch(plt.Rectangle((x, y), w, h, ec="r", lw=2., fc="none"))
ax.imshow(cv2.cvtColor(img_enter, cv2.COLOR_BGR2RGB))

plt.show()