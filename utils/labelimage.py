import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from PIL import Image
from torchvision.transforms import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg


def rotate_image(image):
    rotate_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotate_image


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image


def draw_text_on_image(image, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    font_color = (255, 255, 255)
    position = tuple(map(int, position))

    cv2.putText(image, text, position, font, font_scale, font_color, font_thickness)


def draw_line_union(image, label, mask):
    # h, w, _ = image.shape
    # ------------------------#
    #   将label高度和宽度调整
    #   将label转换为unit8类型的Numpy数组，uint8表示无符号8位整数，取值范围在0-255之间对
    #   label进行阈值处理，>127的像素，值设置为255（实际上是二值化图像）将二值化的label转化为布尔类型的数组
    # label = transforms.Resize((h,w))(label)
    # label = np.uint8(label)
    label = 255 * (label > 0)
    label = label == 255
    print(label.shape)
    print(label.dtype)
    # plt.imshow(label)
    # mask = transforms.Resize((h,w))(mask)
    # mask = np.uint8(mask)
    print(mask.shape)
    print(mask.dtype)
    # plt.imshow(mask)
    # ------------------------------#
    #   cv2.threshold函数对图像进行阈值化处理，>127的像素设置为255
    #   ret_mask是返回的阈值，thresh_mask是返回的阈值化后的图像
    #   cv2.findContours用于找到图像中的轮廓
    #   cv2.RETR_TREE表示检测轮廓时要考虑轮廓的层级结构
    #   cv2.CHAIN_APPROX_SIMPLE表示压缩水平、垂直和对角方向，只保留端点，节省轮廓的存储空间
    #   contours_mask找到的轮廓列表，每个轮廓是一系列点的集合
    #   hierarchy_mask包含了检测到的轮廓之间的关系

    #   为findContours()函数是检测黑底白色对象的轮廓，所以需要转换mask,且要求是单通道的二值图像
    # green = mask[:, :,1] > 0
    # mask[green] = [255,255,255]
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    ret_mask, thresh_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
    plt.imshow(mask)
    # print(mask.dtype)  # uint8
    # print(mask.shape)  # (480, 480, 3) 0-255
    # print(np.max(mask))
    # print(np.min(mask))

    contours_mask, hierarchy_mask = cv2.findContours(thresh_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours_mask, hierarchy_mask = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # --------------------------#
    #   cv2.drawContours将找到的轮廓绘制到图像上
    #   （0，255，0）BGR空间的绿色，2是绘制轮廓的线条宽度
    for cnt_mask in contours_mask:
        cv2.drawContours(image, [cnt_mask], 0, (0, 255, 0), 2)
    h, w = image.shape[:2]
    # plt.figure(w/100, h/100)
    plt.figure(figsize = (w/100, h/100),dpi = 200)
    plt.imshow(image)
    show_mask(label, plt.gca())
    plt.axis('off')
    plt.savefig("output_image.jpg", bbox_inches = 'tight', pad_inches = 0)
    image = cv2.imread("output_image.jpg")
    image = rotate_image(image)
    ## 镜像
    image = cv2.flip(image, 1)
    cv2.imwrite("rotate_image.jpg", image)
    # image = plt.gcf()
    #

    print(image.shape)
    return image


#-----------------#
#   通过cv2.imread读进来的数据被储存为Numpy数组（numpy.ndarray）
#   通过Image.open读进来的数据被储存为PIL库中的Image对象
# image = cv2.imread('./data/0.png')
# label = Image.open("./data/0_gt.jpg")
# mask = Image.open("./data/0_pred.jpg")
if __name__ == "__main__":
    # -----------------------------------------#
    #   读入的时候，image（480，480，3），label（480，480），mask(480，480，3)
    data_path = "D:/UNet_py/dataset_split/npzgood/D1558158_slice006.npz"
    data = np.load(data_path)
    image, label = data['image'], data['label']
    # mask = Image.open("/media/admin2/02CE3F30CE3F1AFD/liwenyi/bishe/seg-models/SCUNet-plusplus-main/predictions/TU_Synapse224_2/output/D1293632_slice005.png")
    mask = cv2.imread(
        "D:/UNet_py/Pytorch-UNet-master/Pytorch-UNet-master/unet/test_predictions/D1558158_slice006.png")

    # print(image.dtype) # uint8
    # print(image.shape) # (480, 480, 3)
    # print(label.dtype) # uint8
    # print(mask.dtype) # uint8
    # print(mask.shape) # (480, 480, 3) 0-255
    # print(np.max(mask))
    # print(np.min(mask))
    # -----------------------------------------#
    draw_line_union(image, label, mask)
