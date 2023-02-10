import albumentations as A
import numpy as np
import onnxruntime as ort
from albumentations.pytorch import ToTensorV2


class MemesClassification:
    def __init__(
        self,
        model_path="./saved_model_memes_classification.onnx",
        image_size=(224, 224),
        device_inference="cpu",
    ):
        self.model_path = model_path
        self.image_size = image_size
        self.device_inference = device_inference
        self.score_threshold = 8.034088775256156e-05
        self.class_names = ["non meme", "meme"]

        self.ort_session = ort.InferenceSession(
            self.model_path,
            providers=[
                "CPUExecutionProvider"
                if device_inference == "cpu"
                else "CUDAExecutionProvider"
            ],
        )

    def predict(self, image):
        preprocessing_image = self.preprocessing_image(image)
        input_onnx = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(
            None,
            {input_onnx: preprocessing_image},
        )
        score = self.postprocessing_output(outputs)
        return self.class_names[int(score > self.score_threshold)]

    def preprocessing_image(self, image_pillow):
        image_augmentation = A.Compose(
            [
                A.Resize(self.image_size[1], self.image_size[0]),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        image = image_augmentation(image=np.array(image_pillow))["image"]
        image = np.expand_dims(image, axis=0)
        return image

    def postprocessing_output(self, output):
        return self.sigmoid(output[0][0][0])

    def sigmoid(self, x):
        return np.exp(-np.logaddexp(0, -x))
