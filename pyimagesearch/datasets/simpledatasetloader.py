import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		self.preprocessors = preprocessors

		if self.preprocessors is None:
			self.preprocessors = []


	def load(self, image_paths, verbose=-1):
		data = []
		labels = []

		for index, image_path in enumerate(image_paths):
			# assuming path : /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(image_path)
			label = image_path.split(os.path.sep)[-2]

			if self.preprocessors is not None:
				for preprocessor in self.preprocessors:
					image = preprocessor.preprocess(image)

			data.append(image)
			labels.append(label)

			if verbose > 0 and index > 0 and (index + 1) % verbose == 0:
				print(f'[INFO] processed {index + 1}/{len(image_paths)}')

		return (np.array(data), np.array(labels))


