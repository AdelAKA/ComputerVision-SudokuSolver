from project.clean import utils as utils
from project.clean.performance import Performance
import cv2 as cv
import numpy as np
from sudoku import Sudoku as s

from project.clean.predictors.neural_network_predictor import NeuralNetworkPredictor


class Sudoku:
    def __init__(self, predictor, debug=True, show_image_before_model_feed=False):
        self.solver = None  # s.Sudoku(3, 3, board=board)
        self.debug = debug
        self.logger = Performance(debug=self.debug)
        self.show_image_before_model_feed = show_image_before_model_feed
        self.predictor = predictor

        self.color_source_image = None
        self.source_image = None
        self.preprocessed_image = None
        self.perspective_array = None
        self.color_sudoku_board = None
        self.sudoku_board = None
        self.sudoku_cells_images = None
        self.sudoku_cells = None

    def pre_process_source_image(self, camera):
        self.logger.tick('pre_process source image')
        src = self.source_image.copy()
        gauss_blur = utils.gaussian_blur(src, (13, 13))
        self.preprocessed_image = utils.adaptive_thresh(gauss_blur)
        cv.imshow("adapt", self.preprocessed_image)
        if camera:
            self.preprocessed_image = cv.dilate(self.preprocessed_image.copy(), (19, 19), iterations=4)
            cv.imshow("dialated", self.preprocessed_image)
        self.logger.end('pre_process source image')

    def get_sudoku_board(self):
        self.logger.tick('get sudoku board')
        src = self.preprocessed_image.copy()
        max_contour = utils.find_biggest_contour(src)
        if max_contour is not None:
            temp = cv.drawContours(self.color_source_image, [max_contour], 0, (0, 255, 0), 3)
            cv.imshow("contour board", temp)
            cnt = cv.approxPolyDP(max_contour, 0.01 * cv.arcLength(max_contour, True), True)
            if len(cnt) == 4:
                self.sudoku_board, self.color_sudoku_board, rect, self.perspective_array = \
                    utils.perspective_transformation(
                        src, self.color_source_image, cnt)

            else:
                self.sudoku_board = None
        else:
            self.sudoku_board = None

        self.logger.end('get sudoku board')

    def get_sudoku_cells_images(self):
        if self.sudoku_board is None:
            return False
        self.sudoku_cells_images = utils.get_numbers_contours(self.sudoku_board.copy(), self.color_sudoku_board.copy())
        self.sudoku_cells = np.zeros((9, 9), dtype=np.uint8)
        self.logger.end('get sudoku cells images')
        return True

    # def pre_process_cell_image(self, src):
    #     print('pre_process_cell_image input is')
    #     print(src)
    #     self.logger.tick('pre process cell image')
    #     src = cv.resize(src, (utils.digit_pic_size, utils.digit_pic_size))
    #     src = np.uint8(src)
    #     # _, src = cv.threshold(src, 100, 255, cv.THRESH_BINARY)
    #     # src = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 2)
    #     # for now otsu is the best trade off between speed and accuracy
    #     _, src = cv.threshold(src, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #     # PS: each pre_processing before largest_connected_component must consider that front object is 1 (white)
    #     # and the background is 0 (black)
    #     contour = utils.find_biggest_contour(src)
    #     src = utils.largest_connected_component(src)
    #     is_number = utils.is_number(src)
    #     if not is_number:
    #         return np.ones((utils.digit_pic_size, utils.digit_pic_size), dtype=np.uint8), is_number
    #     src = np.uint8(src)
    #     padding = 1
    #     x, y, w, h = cv.boundingRect(contour)
    #     cropped = src[y - padding:y + padding + h, x - padding:x + w + padding]
    #     if cropped.size is 0:
    #         return np.ones((utils.digit_pic_size, utils.digit_pic_size), dtype=np.uint8), False
    #     cropped = cv.resize(cropped, (28, 28))
    #     self.logger.end('pre process cell image')
    #     return cropped, is_number,

    def pre_process_cell_image(self, src):
        print('pre_process_cell_image input is')
        self.logger.tick('pre process cell image')
        src = cv.resize(src, (utils.digit_pic_size, utils.digit_pic_size))
        src = np.uint8(src)
        # _, src = cv.threshold(src, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # PS: each pre_processing before largest_connected_component must consider that front object is 1 (white)
        # and the background is 0 (black)

        # cropped = cv.resize(cropped, (28, 28))
        self.logger.end('pre process cell image')
        return None, None

    def predict(self, src):
        self.logger.tick('model predict')
        value = self.predictor.predict(src, show_image_before_model_feed=self.show_image_before_model_feed)
        answer = np.argmax(value[0], axis=0) + 1
        self.logger.end('model predict')
        return answer

    def pre_process_cell_image_and_predict(self, cell):
        self.logger.tick('pre process cell _ image and predict')
        self.sudoku_cells[cell.position[0]][cell.position[1]] = self.predict(cell.image)
        self.logger.end('pre process cell _ image and predict')

    # TODO create 81 thread and each one edit self.sudoku_cells_images in (x,y)
    # TODO consider not create thread to 0 or empty cell for optimise
    def pre_process_cells_images(self):
        self.logger.tick('pre process cells images')
        # for i in range(9):
        #     for j in range(9):
        #         self.pre_process_cell_image_and_predict(i, j)
        for cell_image in self.sudoku_cells_images:
            # print(cell_image.position[0], cell_image.position[1])
            self.pre_process_cell_image_and_predict(cell_image)

        self.logger.end('pre process cells images')

    def show_result_image(self):
        # cv.imshow('source image', self.source_image)
        # cv.imshow('preprocessed image', self.preprocessed_image)
        # print("image h and w", self.sudoku_board.shape[0], self.sudoku_board.shape[1])
        if self.sudoku_board is not None:
            cv.imshow('sudoku board', self.sudoku_board)
        # cv.imshow('sudoku cells', self.sudoku_cells_images[0][0])
        # cv.waitKey(0)

    def feed(self, image, show_result_image=False, real_board=None, camera=False):
        self.logger.tick('frame time', force_print=True)
        self.color_source_image = image
        self.source_image = cv.cvtColor(self.color_source_image, cv.COLOR_BGR2GRAY)
        self.pre_process_source_image(camera)
        self.get_sudoku_board()

        has_board = self.get_sudoku_cells_images()
        if has_board:
            self.pre_process_cells_images()
            # print(self.sudoku_cells)
            self.solve()

        self.logger.end('frame time', force_print=True)
        if self.debug and self.sudoku_cells is not None:
            utils.pretty_model_result(real_board, self.sudoku_cells)
        if show_result_image:
            self.show_result_image()

    def solve(self):
        self.solver = s(width=3, height=3, board=self.sudoku_cells.tolist())
        image_with_solution = utils.write_solution_on_image(self.color_sudoku_board.copy(), self.solver.solve().board,
                                                            self.sudoku_cells)
        # cv.imshow("sol", image_with_solution)
        result_sudoku = cv.warpPerspective(image_with_solution, self.perspective_array,
                                           (self.source_image.shape[1], self.source_image.shape[0])
                                           , flags=cv.WARP_INVERSE_MAP)
        result = np.where(result_sudoku.sum(axis=-1, keepdims=True) != 0, result_sudoku, self.color_source_image)
        cv.imshow('result', result)
        # cv.waitKey(0)


# if __name__ == '__main__':
#     board = [
#         [0, 0, 0, 8, 3, 0, 0, 9, 0],
#         [0, 8, 6, 0, 0, 0, 0, 5, 1],
#         [0, 9, 0, 4, 0, 6, 0, 0, 0],
#         [8, 0, 2, 0, 0, 0, 7, 0, 0],
#         [9, 0, 0, 0, 0, 0, 0, 0, 5],
#         [0, 0, 5, 0, 0, 0, 9, 0, 8],
#         [0, 0, 0, 3, 0, 9, 0, 6, 0],
#         [5, 6, 0, 0, 0, 0, 3, 1, 0],
#         [0, 4, 0, 0, 7, 5, 0, 0, 0]
#     ]
#     model = NeuralNetworkPredictor()
#     sudoku = Sudoku(model, debug=True, show_image_before_model_feed=False)
#     sudoku.feed(cv.imread('resources/sod2.jpg'), real_board=board, show_result_image=False)

# if __name__ == '__main__':
#     board = [
#         [4, 8, 3, 7, 2, 6, 1, 5, 9],
#         [7, 2, 6, 1, 5, 9, 4, 8, 3],
#         [1, 5, 9, 4, 8, 3, 7, 2, 6],
#         [8, 3, 7, 2, 6, 1, 5, 9, 4],
#         [2, 6, 1, 5, 9, 4, 8, 3, 7],
#         [5, 9, 4, 8, 3, 7, 2, 6, 1],
#         [3, 7, 2, 6, 1, 5, 9, 4, 8],
#         [6, 1, 5, 9, 4, 8, 3, 7, 2],
#         [9, 4, 8, 3, 7, 2, 6, 1, 5]
#     ]
#     model = NeuralNetworkPredictor()
#     sudoku = Sudoku(model, debug=True, show_image_before_model_feed=True)
#     sudoku.feed(cv.imread('resources/sod4.jpg'), real_board=board)


# if __name__ == '__main__':
#     board = [
#         [5, 3, 0, 0, 7, 0, 0, 0, 0],
#         [6, 0, 0, 1, 9, 5, 0, 0, 0],
#         [0, 9, 8, 0, 0, 0, 0, 6, 0],
#         [8, 0, 0, 0, 6, 0, 0, 0, 3],
#         [4, 0, 0, 8, 0, 3, 0, 0, 1],
#         [7, 0, 0, 0, 2, 0, 0, 0, 6],
#         [0, 6, 0, 0, 0, 0, 2, 8, 0],
#         [0, 0, 0, 4, 1, 9, 0, 0, 5],
#         [0, 0, 0, 0, 8, 0, 0, 7, 9]
#     ]
#     model = NeuralNetworkPredictor()
#     sudoku = Sudoku(model, debug=False, show_image_before_model_feed=False)
#     # cap = cv.VideoCapture("resources/sudoku2.mp4")
#     cap = cv.VideoCapture(0)
#     cap.set(3, 1280)  # HD Camera
#     cap.set(4, 720)
#     old_sudoku = None
#
#     while True:
#         ret, frame = cap.read()  # Read the frame
#         if ret:
#             frame = cv.resize(frame, (1280 * 3 // 3, 720 * 4 // 3))
#             sudoku.feed(frame, real_board=board, show_result_image=True, camera=True)
#             cv.imshow("frame", frame)
#             if cv.waitKey(1) & 0xFF == ord('q'):  # Hit q if you want to stop the camera
#                 break
#         else:
#             break
#
#     cap.release()
#     # out.release()
#     cv.destroyAllWindows()

if __name__ == '__main__':
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    model = NeuralNetworkPredictor()
    sudoku = Sudoku(model, debug=True, show_image_before_model_feed=False)
    sudoku.feed(cv.imread('resources/sod6.jpg'), real_board=board, show_result_image=False)
    # sudoku.feed(cv.imread('resources/sudoku.jpg'), real_board=board, show_result_image=False)
    cv.waitKey(0)

# if __name__ == '__main__':
#     board = [
#         [0, 0, 0, 6, 0, 4, 7, 0, 0],
#         [7, 0, 6, 0, 0, 0, 0, 0, 9],
#         [0, 0, 0, 0, 0, 5, 0, 8, 0],
#         [0, 7, 0, 0, 2, 0, 0, 9, 3],
#         [8, 0, 0, 0, 0, 0, 0, 0, 5],
#         [4, 3, 0, 0, 1, 0, 0, 7, 0],
#         [0, 5, 0, 2, 0, 0, 0, 0, 0],
#         [3, 0, 0, 0, 0, 0, 2, 0, 8],
#         [0, 0, 2, 3, 0, 1, 0, 0, 0]
#     ]
#     model = NeuralNetworkPredictor()
#     sudoku = Sudoku(model, debug=False, show_image_before_model_feed=True)
#     sudoku.feed(cv.imread('resources/sudoku.jpg'), real_board=board, show_result_image=True)
#     cv.waitKey(0)
