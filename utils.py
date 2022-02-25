import cv2 as cv
import numpy as np

from project.clean.cell_info import CellInfo

digit_pic_size = 28


def gaussian_blur(src, kernel_size=(5, 5), sigmaX=0):
    return cv.GaussianBlur(src, kernel_size, sigmaX)


def cross_closing(src):
    se = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    return cv.morphologyEx(src, cv.MORPH_CLOSE, se)


def disk_opening(src, sz=3):
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (sz, sz))
    return cv.morphologyEx(src, cv.MORPH_OPEN, se)


def disk_closing(src, sz=3):
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (sz, sz))
    return cv.morphologyEx(src, cv.MORPH_CLOSE, se)


def line_opening(src, sz):
    #  self.sudoku_board.shape[1] // 50
    horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (sz, 1))
    return cv.morphologyEx(src, cv.MORPH_CLOSE, horizontal_structure)


def adaptive_thresh(src):
    # return cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C | cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     # cv.THRESH_BINARY_INV,
    #     # 5, 2)
    return cv.adaptiveThreshold(src, 255, 1, 1, 11, 2)  # for threshold and inverse at once


def find_contours(src):
    return cv.findContours(src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


def find_biggest_contour(src):
    contours, hierarchy = find_contours(src)
    if len(contours) != 0:
        return max(contours, key=cv.contourArea)
    pass


def coordinates_sum(corner):
    return corner[0] + corner[1]


def coordinates_division(corner):
    return corner[0] - corner[1]


def perspective_transformation(img, color_image, cnt):
    corners = np.zeros((4, 2), dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    for i in range(4):
        corners[i] = cnt[i][0]

    # Top left
    rect[0] = min(corners, key=coordinates_sum)
    # Bottom right
    rect[2] = max(corners, key=coordinates_sum)
    # Top right
    rect[1] = max(corners, key=coordinates_division)
    # Bottom left
    rect[3] = min(corners, key=coordinates_division)

    (tl, tr, br, bl) = rect
    # the actual width of our Sudoku board
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # the actual height of our Sudoku board
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))
    pts1 = np.float32([rect[0], rect[1], rect[2], rect[3]])
    pts2 = np.float32(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]])  # TL -> TR -> BR -> BL
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(img, M, (max_width, max_height))
    color_dst = cv.warpPerspective(color_image, M, (max_width, max_height))
    return dst, color_dst, rect, M


def crop_image_to_cells(img):
    # _, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
    h = img.shape[0] // 9
    w = img.shape[1] // 9
    offset_w = np.math.floor(w / 10)  # Offset is used to get rid of the boundaries
    offset_h = np.math.floor(h / 10)
    print("===========", img.shape, h, w, offset_h, offset_w)
    blocks = np.zeros(
        (9, 9, h - (offset_h + offset_w), w - (offset_h + offset_w))
    )
    for i in range(9):
        for j in range(9):
            # n = i * 9 + j
            blocks[i][j] = img[h * i + offset_h:h * (i + 1) - offset_h, w * j + offset_w:w * (j + 1) - offset_w]
            if i == 2 and j == 0:
                cv.imshow("croped", blocks[i][j])
            if i == 3 and j == 0:
                cv.imshow("croped1", blocks[i][j])
    return blocks


def write_solution_on_image(source, solution, sudoku_cells):
    # Write grid on image
    SIZE = 9
    width = source.shape[1] // 9
    height = source.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if sudoku_cells[i][j] != 0:  # If user fill this cell
                continue  # Move on
            # if i == 4 and j == 8:
            # print("==================", solution[i][j], i, j)
            text = str(solution[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv.getTextSize(text, font, fontScale=1, thickness=3)
            marginX = np.floor(width / 7)
            marginY = np.floor(height / 7)

            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = int(width * j + np.floor((width - text_width) / 2) + off_set_x)
            bottom_left_corner_y = int(height * (i + 1) - np.floor((height - text_height) / 2) + off_set_y)
            source = cv.putText(source, text, (bottom_left_corner_x, bottom_left_corner_y),
                                font, font_scale, (0, 255, 0), thickness=3, lineType=cv.LINE_AA)
    return source


def is_number(number):
    match = True
    if number.sum() >= digit_pic_size ** 2 * 255 - digit_pic_size * 2 * 255:
        match = False
    else:
        # Criteria 2 for detecting white cell
        # Huge white area in the center
        center_width = number.shape[1] // 2
        center_height = number.shape[0] // 2
        x_start = center_height // 2
        x_end = center_height // 2 + center_height
        y_start = center_width // 2
        y_end = center_width // 2 + center_width
        center_region = number[x_start:x_end, y_start:y_end]
        if center_region.sum() >= center_width * center_height * 255 - 255:
            match = False
    return match


# def is_number(number, i, j):
#     match = True
#     # Criteria 2 for detecting white cell
#     # Huge white area in the center
#     center_width = number.shape[1] // 2
#     center_height = number.shape[0] // 2
#     x_start = center_height // 2
#     x_end = center_height // 2 + center_height
#     y_start = center_width // 2
#     y_end = center_width // 2 + center_width
#     center_region = number[x_start:x_end, y_start:y_end]
#     if center_region.sum() >= center_width * center_height * 255 - 255:
#         match = False
#     return match


def largest_connected_component(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    if len(sizes) <= 1:
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image

    max_label = 1
    # Start from component 1 (not 0) because we want to leave out the background
    max_size = sizes[1]

    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2


def remove_side_lines(img, ratio):
    """
        Remove black lines from image sides
    """

    while np.sum(img[0]) <= (1 - ratio) * img.shape[1] * 255:
        img = img[1:]
    # Bottom
    while np.sum(img[:, -1]) <= (1 - ratio) * img.shape[1] * 255:
        img = np.delete(img, -1, 1)
    # Left
    while np.sum(img[:, 0]) <= (1 - ratio) * img.shape[0] * 255:
        img = np.delete(img, 0, 1)
    # Right
    while np.sum(img[-1]) <= (1 - ratio) * img.shape[0] * 255:
        img = img[:-1]

    return img


def pretty_model_result(real_board, predicted_board):
    right = 0
    wrong = 0
    print('<------------------->')
    for i in range(9):
        for j in range(9):
            score = 'Wrong'
            if real_board[i][j] == 0:
                continue
            if real_board[i][j] == predicted_board[i][j]:
                right += 1
                score = 'Right'
            else:
                wrong += 1
            print("{} predict {} for {} position({},{})".format(score, predicted_board[i][j], real_board[i][j], i + 1,
                                                                j + 1))
    print('<------------------->')
    print("{} Right predict and {} Wrong predict".format(right, wrong))
    print('<------------------->\n')


def get_numbers_contours(sudoku_board, colored_sudoku_board):
    cv.imshow("main", sudoku_board)
    # temp = gaussian_blur(sudoku_board, (3,3))
    # cv.imshow("temp", temp)sudoku_board
    # temp = cv.adaptiveThreshold(temp, 255,
    #                      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                      cv.THRESH_BINARY,
    #                      11, 2)

    # temp = adaptive_thresh(temp)
    # temp = cv.bitwise_not(temp)
    # cv.imshow("temp2", temp)
    # temp = disk_opening(sudoku_board)
    # temp = disk_closing(temp)
    # erd = cv.erode(disk_open, kernel=(3, 3))
    # dil = cv.dilate(erd, kernel=(3, 3))

    # cv.imshow("temp3", temp)
    # mask = largest_connected_component(
    #     cv.dilate(sudoku_board, kernel=(3, 3)))  # dilate to fully connect the board lines
    mask = largest_connected_component(disk_closing(sudoku_board))
    cv.imshow("mask", mask)
    numbers = np.clip(sudoku_board * (mask[:, :] / 255), 0, 255).astype(np.uint8)
    lol = disk_opening(numbers)
    # lol = cv.bitwise_not(lol)
    # lol = gaussian_blur(lol, (3, 3))
    # lol = adaptive_thresh(lol)
    # _, lol = cv.threshold(lol, 150, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # lol = disk_opening(lol)
    cv.imshow("lol", lol)
    contours, hierarchy = cv.findContours(lol, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda con: cv.contourArea(con), reverse=True)
    padding = 4

    #cv.waitKey(0)
    i = 0
    numbers = []
    valid_area_ratio = 0.058
    max_h_to_w_ratio = 75
    min_h_to_w_ratio = 30
    # print("the allowed area", sudoku_board.size * valid_area_ratio / 100)
    image_h_to_w_ratio = sudoku_board.shape[1] / sudoku_board.shape[0] * 3
    for c in contours:
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)
        image_to_cell_ratio = (h / w) * 100 / image_h_to_w_ratio
        if area * 100 < sudoku_board.size * valid_area_ratio \
                or max_h_to_w_ratio < image_to_cell_ratio or image_to_cell_ratio < min_h_to_w_ratio:
            # print(area, "not valid")
            continue
        else:
            # print(area, "is valid")
            numbers.append(
                CellInfo(sudoku_board[y - padding:y + padding + h, x - padding:x + w + padding], [x, y],
                         sudoku_board.shape)
            )
        #     cv.imshow("number" + str(i), sudoku_board[y - padding:y + padding + h, x - padding:x + w + padding])
        #     i += 1
        # bom = cv.drawContours(colored_sudoku_board, [c], 0, (0, 255, 0), 3)
        # cv.imshow("cont", bom)
        # cv.waitKey(0)
    return numbers
