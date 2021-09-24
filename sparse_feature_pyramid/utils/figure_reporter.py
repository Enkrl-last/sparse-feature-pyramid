# import matplotlib.pyplot as plt
# import numpy as np
#
#
# class FigureReporter(object):
#     def report_figures(self, model, batch, output):
#         pass
#
#     def report_reconstructed_image(self, input_image, output_image):
#         fig, ax = plt.subplots(1, 2, dpi=200)
#         output_image = output_image[0][0].detach().cpu().numpy().transpose(1, 2, 0)
#         output_image = output_image * np.array(data_module._std)[None, None] + np.array(data_module._mean)[None, None]
#         output_image = np.clip(output_image, 0, 1)
#         ax[1].imshow(output_image)
#
#         input_image = torch.nn.functional.interpolate(batch["image"], size=output_image.shape[:2])
#         input_image = input_image[0].detach().cpu().numpy().transpose(1, 2, 0)
#         input_image = input_image * np.array(data_module._std)[None, None] + np.array(data_module._mean)[None, None]
#         input_image = np.clip(input_image, 0, 1)
#         ax[0].imshow(input_image)
#
#     def report_probabilities(self):
#         pass
#
#     def report_mask(self):
#         pass
#
#     def report_level_reconstruction(self):
