import matplotlib.pyplot as plt
import numpy as np
import clearml
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class ClearmlFigureReporter(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._image_index = 0
        self._logger = clearml.Logger.current_logger()

    def report_figures(self, input_image, output_image, masks, probabilities, iteration):
        reconstructed_image_figure = self.reconstructed_image_figure(input_image, output_image)
        self.report_matplotlib_figure("reconstructed_image", f"{self._image_index}",
                                      reconstructed_image_figure, iteration)
        self.report_matplotlib_figure("predicted_mask", f"{self._image_index}",
                                      self.predicted_mask_figure(masks), iteration)
        self.report_matplotlib_figure("predicted_probabilities", f"{self._image_index}",
                                      self.predicted_mask_figure(probabilities), iteration)

    def report_matplotlib_figure(self, title, series, figure, iteration):
        canvas = FigureCanvas(figure)
        canvas.draw()
        image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        self._logger.report_image(title, series, iteration, image=image)

    def reconstructed_image_figure(self, input_image, output_image):
        figure, ax = plt.subplots(1, 2, dpi=200)
        ax[0].imshow(self.prepare_image(input_image))
        ax[1].imshow(self.prepare_image(output_image))
        return figure

    def prepare_image(self, image):
        image = image[self._image_index].detach().cpu().numpy().transpose(1, 2, 0)
        image = image * np.array(self._std)[None, None] + np.array(self._mean)[None, None]
        image = np.clip(image, 0, 1)
        return image

    def predicted_mask_figure(self, masks):
        fig, ax = plt.subplots(1, len(masks), dpi=200)
        for i in range(len(masks)):
            mask = masks[i][self._image_index][0].detach().cpu().numpy()
            ax[i].imshow(mask, cmap="gray")
        return fig
