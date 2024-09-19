import os

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageCms, ImageOps, ImageDraw, ImageFont

sRGB_profile = ImageCms.createProfile("sRGB")
Lab_profile = ImageCms.createProfile("LAB")


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def adjust_shadows(luminance_array, shadow_intensity, hdr_intensity):
    # Darken shadows more as shadow_intensity increases, scaled by hdr_intensity
    return np.clip(luminance_array - luminance_array * shadow_intensity * hdr_intensity * 0.5, 0, 255)


def adjust_highlights(luminance_array, highlight_intensity, hdr_intensity):
    # Brighten highlights more as highlight_intensity increases, scaled by hdr_intensity
    return np.clip(luminance_array + (255 - luminance_array) * highlight_intensity * hdr_intensity * 0.5, 0, 255)


def apply_adjustment(base, factor, intensity_scale):
    """Apply positive adjustment scaled by intensity."""
    # Ensure the adjustment increases values within [0, 1] range, scaling by intensity
    adjustment = base + (base * factor * intensity_scale)
    # Ensure adjustment stays within bounds
    return np.clip(adjustment, 0, 1)


def multiply_blend(base, blend):
    """Multiply blend mode."""
    return np.clip(base * blend, 0, 255)


def overlay_blend(base, blend):
    """Overlay blend mode."""
    # Normalize base and blend to [0, 1] for blending calculation
    base = base / 255.0
    blend = blend / 255.0
    return np.where(base < 0.5, 2 * base * blend, 1 - 2 * (1 - base) * (1 - blend)) * 255


def adjust_shadows_non_linear(luminance, shadow_intensity, max_shadow_adjustment=1.5):
    lum_array = np.array(luminance, dtype=np.float32) / 255.0  # Normalize
    # Apply a non-linear darkening effect based on shadow_intensity
    shadows = lum_array ** (1 / (1 + shadow_intensity * max_shadow_adjustment))
    return np.clip(shadows * 255, 0, 255).astype(np.uint8)  # Re-scale to [0, 255]


def adjust_highlights_non_linear(luminance, highlight_intensity, max_highlight_adjustment=1.5):
    lum_array = np.array(luminance, dtype=np.float32) / 255.0  # Normalize
    # Brighten highlights more aggressively based on highlight_intensity
    highlights = 1 - (1 - lum_array) ** (1 + highlight_intensity * max_highlight_adjustment)
    return np.clip(highlights * 255, 0, 255).astype(np.uint8)  # Re-scale to [0, 255]


def merge_adjustments_with_blend_modes(luminance, shadows, highlights, hdr_intensity, shadow_intensity,
                                       highlight_intensity):
    # Ensure the data is in the correct format for processing
    base = np.array(luminance, dtype=np.float32)

    # Scale the adjustments based on hdr_intensity
    scaled_shadow_intensity = shadow_intensity ** 2 * hdr_intensity
    scaled_highlight_intensity = highlight_intensity ** 2 * hdr_intensity

    # Create luminance-based masks for shadows and highlights
    shadow_mask = np.clip((1 - (base / 255)) ** 2, 0, 1)
    highlight_mask = np.clip((base / 255) ** 2, 0, 1)

    # Apply the adjustments using the masks
    adjusted_shadows = np.clip(base * (1 - shadow_mask * scaled_shadow_intensity), 0, 255)
    adjusted_highlights = np.clip(base + (255 - base) * highlight_mask * scaled_highlight_intensity, 0, 255)

    # Combine the adjusted shadows and highlights
    adjusted_luminance = np.clip(adjusted_shadows + adjusted_highlights - base, 0, 255)

    # Blend the adjusted luminance with the original luminance based on hdr_intensity
    final_luminance = np.clip(base * (1 - hdr_intensity) + adjusted_luminance * hdr_intensity, 0, 255).astype(np.uint8)

    return Image.fromarray(final_luminance)


def apply_gamma_correction(lum_array, gamma):
    """
    Apply gamma correction to the luminance array.
    :param lum_array: Luminance channel as a NumPy array.
    :param gamma: Gamma value for correction.
    """
    if gamma == 0:
        return np.clip(lum_array, 0, 255).astype(np.uint8)

    epsilon = 1e-7  # Small value to avoid dividing by zero
    gamma_corrected = 1 / (1.1 - gamma)
    adjusted = 255 * ((lum_array / 255) ** gamma_corrected)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


# create a wrapper function that can apply a function to multiple images in a batch while passing all other arguments to the function
def apply_to_batch(func):
    def wrapper(self, image, *args, **kwargs):
        images = []
        for img in image:
            images.append(func(self, img, *args, **kwargs))
        batch_tensor = torch.cat(images, dim=0)
        return (batch_tensor,)

    return wrapper


# 领域降噪
def calculate_noise_count(img_obj, w, h, width, height):
    count = 0
    for _w_ in [w - 1, w, w + 1]:
        for _h_ in [h - 1, h, h + 1]:
            if _w_ > width - 1:
                continue
            if _h_ > height - 1:
                continue
            if _w_ == w and _h_ == h:
                continue
            if img_obj[_w_, _h_] < 230:  # 这里因为是灰度图像，设置小于230为非白色
                count += 1
    return count


# 领域降噪
def row_noise(pim, height, weight, w):
    for h in range(height):
        if calculate_noise_count(pim, w, h, weight, height) < 4:
            pim[w, h] = 255


def mexx_image_filter(img, image_filter,
                      write_text = '画画的Baby',
                      font_name = 'YangRenDongZhuShiTi-Light-2.ttf',
                      padding_height = 120,
                      font_size = 80):
    if image_filter == "线稿-LINE0":
        # 转换为灰度图
        gray_image = img.convert("L")
        array = np.array(gray_image).astype(np.float32)
        # 根据灰度变化来模拟人类视觉的明暗程度
        depth = 10  # 预设虚拟深度值为10 范围为0-100
        # 提取x y方向梯度值 解构赋给grad_x, grad_y
        grad_x, grad_y = np.gradient(array)
        # 利用像素之间的梯度值和虚拟深度值对图像进行重构
        grad_x = grad_x * depth / 100
        grad_y = grad_y * depth / 100
        # 梯度归一化 定义z深度为1.  将三个梯度绝对值转化为相对值，在三维中是相对于斜对角线A的值
        dis = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.0)
        uni_x = grad_x / dis
        uni_y = grad_y / dis
        uni_z = 1.0 / dis
        # 光源俯视角度和光源方位角度
        vec_el = np.pi / 2.2
        vec_az = np.pi / 4
        # 光源对x、y、z轴的影响
        dx = np.cos(vec_el) * np.cos(vec_az)
        dy = np.cos(vec_el) * np.sin(vec_az)
        dz = np.sin(vec_el)
        # 光源归一化
        out = 255 * (uni_x * dx + uni_y * dy + uni_z * dz)
        out = out.clip(0, 255)
        img = Image.fromarray(out.astype(np.uint8))
        return img.convert("RGB")
    if image_filter == "线稿-LINE1":
        gray_image = img.convert("L")
        im = gray_image.filter(ImageFilter.GaussianBlur(radius=0.75))  # 高斯模糊 75%
        a = np.asarray(im).astype(np.float32)

        depth = 10.  # 设定虚拟深度
        grad = np.gradient(a)
        grad_x, grad_y = grad
        grad_x = grad_x * depth / 100.
        grad_y = grad_y * depth / 100.
        # 梯度向量计算
        A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
        uni_x = grad_x / A
        uni_y = grad_y / A
        uni_z = 1. / A
        vec_el = np.pi / 2.2
        vec_az = np.pi / 4.
        dx = np.cos(vec_el) * np.cos(vec_az)
        dy = np.cos(vec_el) * np.sin(vec_az)
        dz = np.sin(vec_el)

        b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
        b = b.clip(0, 255)  # 二值化处理，要么为0，（黑色边缘）要么为255（白色背景）
        im2 = Image.fromarray(b.astype(np.uint8))
        im2 = im2.filter(ImageFilter.SHARPEN)
        weight, height = im2.size
        # 降噪
        pim = im2.load()
        map(row_noise, [(pim, height, weight, w,) for w in range(weight)])
        # 填充新图像的像素数据
        for i in range(im2.size[0]):
            for j in range(im2.size[1]):
                im2.putpixel((i, j), pim[i, j])
        return im2.convert("RGB")
    if image_filter == "线稿-LINE2":
        # 转换为灰度图
        gray_image = img.convert("L")
        # 应用高斯模糊（可选，根据需要决定是否使用）
        blurred_image = gray_image.filter(ImageFilter.GaussianBlur(0.75))
        # 应用边缘检测
        edge_image = blurred_image.filter(ImageFilter.FIND_EDGES)
        # 锐化边缘
        edge_enhance_image2 = edge_image.filter(ImageFilter.EDGE_ENHANCE)
        inverted_image = ImageOps.invert(edge_enhance_image2)
        return inverted_image.convert("RGB")
    if image_filter == "线稿-LINE3":
        # 转换为灰度图
        gray_image = img.convert("L")
        # 应用高斯模糊（可选，根据需要决定是否使用）
        blurred_image = gray_image.filter(ImageFilter.GaussianBlur(0.75))
        # 应用边缘检测
        edge_image = blurred_image.filter(ImageFilter.CONTOUR)
        # 锐化边缘
        sharpen_image2 = edge_image.filter(ImageFilter.SHARPEN)
        return sharpen_image2.convert("RGB")
    if image_filter == "线稿-LINE3.1":
        # 转换为灰度图
        gray_image = img.convert("L")
        # 应用高斯模糊（可选，根据需要决定是否使用）
        blurred_image = gray_image.filter(ImageFilter.GaussianBlur(0.75))
        # 应用边缘检测
        edge_image = blurred_image.filter(ImageFilter.CONTOUR)
        # 创建边缘图像，例如使用边缘检测滤镜
        # 定义一个锐化的卷积核
        kernel = (-1, -1, -1,
                  -1, 9, -1,
                  -1, -1, -1)
        # 创建自定义滤镜
        custom_filter = ImageFilter.Kernel((3, 3), kernel)
        return edge_image.filter(custom_filter).convert("RGB")
    if image_filter == "线稿-LINE3.2":
        # 转换为灰度图
        gray_image = img.convert("L")
        # 应用高斯模糊（可选，根据需要决定是否使用）
        blurred_image = gray_image.filter(ImageFilter.GaussianBlur(0.75))
        # 应用边缘检测
        edge_image = blurred_image.filter(ImageFilter.FIND_EDGES)
        # 创建边缘图像，例如使用边缘检测滤镜
        blurred_image2 = edge_image.filter(ImageFilter.SMOOTH_MORE)
        inverted_image = ImageOps.invert(blurred_image2)
        return inverted_image.filter(ImageFilter.EDGE_ENHANCE).convert("RGB")
    if image_filter == "线稿-LINE4":
        # 转换为灰度图
        gray_image = img.convert("L")
        # 应用边缘检测
        edge_image = gray_image.filter(ImageFilter.CONTOUR)
        # 锐化边缘
        edge_enhance_image2 = edge_image.filter(ImageFilter.EDGE_ENHANCE)
        return edge_enhance_image2.convert("RGB")
    if image_filter == "线稿-LINE5":
        # 转换为灰度图
        gray_image = img.convert("L")
        # 应用边缘检测
        edge_image = gray_image.filter(ImageFilter.CONTOUR)
        # 对比度
        contrast_enhancer = ImageEnhance.Sharpness(edge_image)
        enhanced_image = contrast_enhancer.enhance(1.2)
        return enhanced_image.convert("RGB")
    elif image_filter == "边缘检测-FIND_EDGES":
        return img.filter(ImageFilter.FIND_EDGES)
    elif image_filter == "轮廓-CONTOUR":
        return img.filter(ImageFilter.CONTOUR)
    elif image_filter == "灰度-L":
        gray_image = img.convert("L")
        return gray_image.convert("RGB")
    elif image_filter == "锐化-SHARPEN":
        return img.filter(ImageFilter.SHARPEN)
    elif image_filter == "锐化-UNSHARP_MASK":
        return img.filter(ImageFilter.UnsharpMask(2))
    elif image_filter == "边缘增强-EDGE_ENHANCE":
        return img.filter(ImageFilter.EDGE_ENHANCE)
    elif image_filter == "边缘增强-EDGE_ENHANCE_MORE":
        return img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif image_filter == "浮雕-EMBOSS":
        return img.filter(ImageFilter.EMBOSS)
    elif image_filter == "平滑-SMOOTH":
        return img.filter(ImageFilter.SMOOTH)
    elif image_filter == "平滑-SMOOTH_MORE":
        return img.filter(ImageFilter.SMOOTH_MORE)
    elif image_filter == "细节-DETAIL":
        return img.filter(ImageFilter.DETAIL)
    elif image_filter == "模糊-BLUR":
        return img.filter(ImageFilter.BLUR)
    elif image_filter == "模糊-BOX_BLUR":
        return img.filter(ImageFilter.BoxBlur(2))
    elif image_filter == "模糊-GAUSSIAN_BLUR":
        return img.filter(ImageFilter.GaussianBlur(0.75))
    elif image_filter == "反相-INVERT":
        return ImageOps.invert(img)
    elif image_filter == "去燥-中值滤波器":
        return img.filter(ImageFilter.MedianFilter(size=3))
    elif image_filter == "翻转_FLIP_LEFT_RIGHT":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif image_filter == "翻转_FLIP_TOP_BOTTOM":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif image_filter == "旋转_ROTATE_45":
        return img.rotate(45)
    elif image_filter == "旋转_ROTATE_90":
        return img.rotate(90)
    elif image_filter == "旋转_ROTATE_180":
        return img.rotate(180)
    elif image_filter == "旋转_ROTATE_270":
        return img.rotate(270)
    elif image_filter == "对比度_0.8":
        contrast_enhancer = ImageEnhance.Contrast(img)
        enhanced_image = contrast_enhancer.enhance(0.8)
        return enhanced_image
    elif image_filter == "对比度_1.2":
        contrast_enhancer = ImageEnhance.Contrast(img)
        enhanced_image = contrast_enhancer.enhance(1.2)
        return enhanced_image
    elif image_filter == "对比度_1.5":
        contrast_enhancer = ImageEnhance.Contrast(img)
        enhanced_image = contrast_enhancer.enhance(1.5)
        return enhanced_image
    elif image_filter == "对比度_2.0":
        contrast_enhancer = ImageEnhance.Contrast(img)
        enhanced_image = contrast_enhancer.enhance(2.0)
        return enhanced_image
    elif image_filter == "对比度_3.0":
        contrast_enhancer = ImageEnhance.Contrast(img)
        enhanced_image = contrast_enhancer.enhance(3.0)
        return enhanced_image
    elif image_filter == "对比度_5.0":
        contrast_enhancer = ImageEnhance.Contrast(img)
        enhanced_image = contrast_enhancer.enhance(5.0)
        return enhanced_image
    elif image_filter == "框":
        border_size = 20
        # 获取原始图片的尺寸
        img_width, img_height = img.size

        # 计算相框的尺寸（原始图片的尺寸加上边框的厚度）
        frame_width = img_width + 2 * border_size
        frame_height = img_height + 2 * border_size

        # 创建一个新的图像对象作为相框
        frame = Image.new('RGB', (frame_width, frame_height), 'white')  # 相框背景为白色
        draw = ImageDraw.Draw(frame)  # 创建绘图对象

        # 将原始图片粘贴到相框中（居中）
        x = border_size
        y = border_size
        frame.paste(img, (x, y))

        return frame.convert("RGB")
    elif image_filter == "留白":
        text = write_text
        text_color = "black"
        # 获取原始图片的尺寸
        img_width, img_height = img.size

        # 创建一个新的图像对象，宽度和高度分别为原始图片宽度和原始高度加上留白高度
        new_height = img_height + padding_height
        new_img = Image.new('RGB', (img_width, new_height), 'white')  # 新图像背景为白色

        # 将原始图片粘贴到新图像上（留白在上方）
        new_img.paste(img, (0, padding_height))

        # 准备绘制文字
        draw = ImageDraw.Draw(new_img)

        current_directory = os.path.dirname(os.path.realpath(__file__))
        font_file = current_directory + "/font/" + font_name

        font = ImageFont.truetype(font_file, font_size)

        # 计算文字的位置（位于留白区域的中间）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (img_width - text_width) / 2
        text_y = (text_height - padding_height) / 8

        # 在留白区域写上文字
        draw.text((text_x, text_y), text, fill=text_color, font=font)

        return new_img.convert("RGB")
    return img


class PilEffects:
    @classmethod
    def INPUT_TYPES(self):
        list = ["NO",
                "线稿-LINE0", "线稿-LINE1", "线稿-LINE2", "线稿-LINE3", "线稿-LINE3.1", "线稿-LINE3.2", "线稿-LINE4",
                "线稿-LINE5",
                "边缘检测-FIND_EDGES", "轮廓-CONTOUR", "灰度-L",
                "细节-DETAIL",
                "平滑-SMOOTH", "平滑-SMOOTH_MORE",
                "锐化-SHARPEN", "锐化-UNSHARP_MASK",
                "边缘增强-EDGE_ENHANCE", "边缘增强-EDGE_ENHANCE_MORE",
                "模糊-BLUR", "模糊-BOX_BLUR", "模糊-GAUSSIAN_BLUR",
                "反相-INVERT",
                "去燥-中值滤波器",
                "浮雕-EMBOSS",
                "翻转_FLIP_LEFT_RIGHT", "翻转_FLIP_TOP_BOTTOM",
                "旋转_ROTATE_45", "旋转_ROTATE_90", "旋转_ROTATE_180", "旋转_ROTATE_270",
                "对比度_0.8", "对比度_1.2", "对比度_1.5", "对比度_2.0", "对比度_3.0", "对比度_5.0",
                "框"
                ]
        return {'required': {'image': ('IMAGE', {'default': None}),
                             "image_filter": (list, {"default": "NO"})
                             }}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('result_img',)
    FUNCTION = 'apply_pil1'
    CATEGORY = 'ComfyUI_Mexx'

    @apply_to_batch
    def apply_pil1(self, image, image_filter="NO", write_text="画画的Baby"):
        # Load the image
        img = tensor2pil(image)
        result_img = mexx_image_filter(img, image_filter, write_text)
        return pil2tensor(result_img)


class PilTitle:
    @classmethod
    def INPUT_TYPES(self):
        list = ["NO",
                "留白"
                ]
        font_names = ["YangRenDongZhuShiTi-Light-2.ttf",
                      "JinNianYeYaoJiaYouYa-2.ttf",
                      "CEFFontsCJKMono-Regular.ttf",
                      "庞门正道粗书体6.0.ttf",
                      "白路飞云手写体.ttf"
                      ]
        return {'required': {'image': ('IMAGE', {'default': None}),
                             "image_filter": (list, {"default": "NO"}),
                             "padding_height": ("INT", {"default": 100}),
                             "font_size": ("INT", {"default": 80}),
                             "font_name": (font_names, {"default": "YangRenDongZhuShiTi-Light-2.ttf"}),
                             "write_text": ("STRING", {"default": "画画的Baby", "multiline": True}),
                             }}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('result_img',)
    FUNCTION = 'apply_pil2'
    CATEGORY = 'ComfyUI_Mexx'

    @apply_to_batch
    def apply_pil2(self, image, image_filter="NO",
                   write_text="画画的Baby",
                   font_name = 'YangRenDongZhuShiTi-Light-2.ttf',
                   padding_height = 120,
                   font_size = 80):
        # Load the image
        img = tensor2pil(image)
        result_img = mexx_image_filter(img, image_filter, write_text, font_name, padding_height, font_size)
        return pil2tensor(result_img)


class PilMergeImage:
    @classmethod
    def INPUT_TYPES(self):
        merge_type_list = ["上下", "左右", "平行四边形"]
        return {'required': {'merge_type': (merge_type_list, {"default": "左右"}),
                             'image': ('IMAGE', {'default': None}),
                             'image2': ('IMAGE', {'default': None}),
                             },
                'optional': {
                    'image3': ('IMAGE', {'default': None}),
                    'image4': ('IMAGE', {'default': None}),
                    'image5': ('IMAGE', {'default': None}),
                    'image6': ('IMAGE', {'default': None}),
                }}

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('result_img',)
    FUNCTION = 'apply_pil3'
    CATEGORY = 'ComfyUI_Mexx'

    @apply_to_batch
    def apply_pil3(self, image, image2, image3 = None, image4  = None, image5  = None, image6 = None, merge_type="左右"):
        print(f"PIL Merge Image Start")
        images = []
        # Load the image
        img1 = tensor2pil(image)
        images.append(img1)
        print(f"Image1: {img1.size}")
        img2 = tensor2pil(image2)
        images.append(img2)
        print(f"Image2: {img2.size}")
        if image3 is not None:
            img3 = tensor2pil(image3)
            images.append(img3)
            print(f"Image3: {img3.size}")
        if image4 is not None:
            img4 = tensor2pil(image4)
            images.append(img4)
            print(f"Image4: {img4.size}")
        if image5 is not None:
            img5 = tensor2pil(image5)
            images.append(img5)
            print(f"Image5: {img5.size}")
        if image6 is not None:
            img6 = tensor2pil(image6)
            images.append(img6)
            print(f"Image6: {img6.size}")

        # 确保所有图片尺寸相同
        for img in images:
            if img.size != images[0].size:
                raise ValueError("All images must be the same size")

        # 获取图片的宽度和高度
        width, height = images[0].size
        print(f"PIL Merge Image: {width}, {height}")

        if merge_type == "左右":
            # 计算每张图片的宽度部分
            part_width = width // len(images)
            print(f"PIL Merge part_width: {part_width}")

            # 创建一个新的图片，用于合成
            combined_image = Image.new('RGB', (width, height))

            # 依次将每张图片的相应部分粘贴到合成图片上
            for i, image in enumerate(images):
                part = image.crop((i * part_width, 0, (i + 1) * part_width, height))
                combined_image.paste(part, (i * part_width, 0))
                print(f"PIL Merge i = {i}")

            print(f"result_img: {combined_image.size}")
            return pil2tensor(combined_image)
        if merge_type == "上下":

            # 计算每张图片的宽度部分
            part_height = height // len(images)
            # 创建一个新的图片，用于合成
            combined_image = Image.new('RGB', (width, height))

            # 依次将每张图片的相应部分粘贴到合成图片上
            for i, image in enumerate(images):
                part = image.crop((0, i * part_height, width, (i + 1) * part_height))
                combined_image.paste(part, (0, i * part_height))
                print(f"PIL Merge i = {i}")

            print(f"result_img: {combined_image.size}")
            return pil2tensor(combined_image)

        if merge_type == "平行四边形":
            # 定义平行四边形的四个顶点坐标（相对于图片左上角）
            # 例如，这里定义了一个从左上角开始的平行四边形
            points = [
                (0, 0),
                (img1.width // 2, 0),
                (img1.width, img1.height),
                (img1.width // 2, img1.height)
                ]

            # 创建一个和图片同样大小的透明遮罩
            mask = Image.new('L', img1.size, 0)
            draw = ImageDraw.Draw(mask)

            # 绘制平行四边形遮罩
            draw.polygon(points, fill=255)

            # 将图2的平行四边形区域应用遮罩
            img2 = img2.copy()
            img2.putalpha(mask)

            # 将图1和图2合成
            combined_image = Image.composite(img2, img1, mask)
            return pil2tensor(combined_image)

        raise ValueError("Merge Type Error")

def adjust_brightness(image, brightness_factor):
    enhancer = ImageEnhance.Brightness(image)
    adjusted_image = enhancer.enhance(brightness_factor)
    return adjusted_image


def calculate_brightness_factor(target_brightness, current_brightness):
    return target_brightness / current_brightness


def get_average_brightness(image):
    grayscale_image = image.convert("L")
    histogram = grayscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    total_brightness = sum(i * w for i, w in enumerate(histogram))
    return total_brightness / pixels


def apply_dithering(image):
    return image.convert("P", palette=Image.ADAPTIVE, colors=256).convert("RGB")


def apply_noise_reduction(image, strength):
    return image.filter(ImageFilter.GaussianBlur(radius=strength))


def apply_gradient_smoothing(image, strength):
    return image.filter(ImageFilter.SMOOTH_MORE if strength > 1 else ImageFilter.SMOOTH)


def blend_images(image1, image2, alpha):
    return Image.blend(image1, image2, alpha)


def temporal_smoothing(frames, window_size):
    num_frames = len(frames)
    smoothed_frames = []

    for i in range(num_frames):
        start = max(0, i - window_size // 2)
        end = min(num_frames, i + window_size // 2 + 1)
        window_frames = frames[start:end]

        smoothed_frame = np.mean(window_frames, axis=0)
        smoothed_frames.append(smoothed_frame)

    return smoothed_frames


def resize_and_crop(pil_img, target_width, target_height):
    """Resize and crop an image to fit exactly the specified dimensions."""
    original_width, original_height = pil_img.size
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if target_aspect_ratio > aspect_ratio:
        # Target is wider than the image
        scale_factor = target_width / original_width
        scaled_height = int(original_height * scale_factor)
        scaled_width = target_width
    else:
        # Target is taller than the image
        scale_factor = target_height / original_height
        scaled_height = target_height
        scaled_width = int(original_width * scale_factor)

    # Resize the image
    resized_img = pil_img.resize((scaled_width, scaled_height), Image.BILINEAR)

    # Crop the image
    if scaled_width != target_width or scaled_height != target_height:
        left = (scaled_width - target_width) // 2
        top = (scaled_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        cropped_img = resized_img.crop((left, top, right, bottom))
    else:
        cropped_img = resized_img

    return cropped_img


NODE_CLASS_MAPPINGS = {
    'PIL Effects (Mexx)': PilEffects,
    'PIL TITLE (Mexx)': PilTitle,
    'PIL Merge Image (Mexx)': PilMergeImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'PIL_Effects': 'PIL Effects (Mexx)',
    'PIL_TITLE': 'PIL TITLE (Mexx)',
    'PIL_MergeImage': 'PIL Merge Image (Mexx)'
}
