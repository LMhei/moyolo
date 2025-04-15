# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
)
from ultralytics.utils import ROOT, yaml_load


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """
        初始化YOLO模型。

该构造函数用于初始化YOLO模型，会自动根据模型文件名切换到特定模型类型
(如YOLOWorld或YOLOE)。

Args:
    model (str | Path): 模型名称或模型文件路径，例如：'yolo11n.pt', 'yolov8n.yaml'
    task (str | None): YOLO任务类型，可选值：'detect'(检测), 'segment'(分割), 'classify'(分类),
                    'pose'(姿态), 'obb'(定向边界框)。默认为基于模型的自动检测
    verbose (bool): 加载时是否显示模型信息

Examples:
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolov8n.pt")  # 加载预训练的YOLOv8n检测模型
    >>> model = YOLO("yolov8n-seg.pt")  # 加载预训练的YOLOv8n分割模型
    >>> model = YOLO("yolo11n.pt")  # 加载预训练的YOLOv11n检测模型
        """
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOE PyTorch model
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            # ------------------------- 分类任务映射 -------------------------
            "classify": {
                "model": ClassificationModel,  # 图像分类模型架构
                "trainer": yolo.classify.ClassificationTrainer,  # 分类训练流程控制
                "validator": yolo.classify.ClassificationValidator,  # 分类精度验证逻辑
                "predictor": yolo.classify.ClassificationPredictor,  # 分类推理接口
            },

            # ------------------------- 目标检测任务映射 ---------------------
            "detect": {
                "model": DetectionModel,  # YOLO检测核心网络结构
                "trainer": yolo.detect.DetectionTrainer,  # 检测任务训练器（包含数据增强/损失计算等）
                "validator": yolo.detect.DetectionValidator,  # mAP等检测指标计算
                "predictor": yolo.detect.DetectionPredictor,  # 检测推理实现（支持图片/视频/流媒体）
            },

            # ------------------------- 实例分割任务映射 ---------------------
            "segment": {
                "model": SegmentationModel,  # 分割模型（检测头+掩膜预测）
                "trainer": yolo.segment.SegmentationTrainer,  # 分割专用训练流程
                "validator": yolo.segment.SegmentationValidator,  # 分割精度评估（box/mask mAP）
                "predictor": yolo.segment.SegmentationPredictor,  # 分割结果生成（输出掩膜数据）
            },

            # ------------------------- 姿态估计任务映射 ---------------------
            "pose": {
                "model": PoseModel,  # 关键点检测网络结构
                "trainer": yolo.pose.PoseTrainer,  # 姿态估计训练逻辑
                "validator": yolo.pose.PoseValidator,  # 关键点精度验证（OKS指标）
                "predictor": yolo.pose.PosePredictor,  # 实时姿态预测接口
            },

            # ------------------------- 定向边界框任务映射 -------------------
            "obb": {
                "model": OBBModel,  # 旋转框检测模型（基于角度预测）
                "trainer": yolo.obb.OBBTrainer,  # 旋转框专用训练器
                "validator": yolo.obb.OBBValidator,  # 旋转框IoU计算
                "predictor": yolo.obb.OBBPredictor,  # 旋转框预测结果生成
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set the model's class names for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """YOLOE object detection and segmentation model."""

    def __init__(self, model="yoloe-v8s-seg.pt", task=None, verbose=False) -> None:
        """
        Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """
        Get visual positional embeddings for the given image and visual features.

        This method extracts positional embeddings from visual features based on the input image. It requires
        that the model is an instance of YOLOEModel.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features extracted from the image.

        Returns:
            (torch.Tensor): Visual positional embeddings.

        Examples:
            >>> model = YOLOE("yoloe-v8s.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = model.model.backbone(img)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab, names):
        """
        Set vocabulary and class names for the YOLOE model.

        This method configures the vocabulary and class names used by the model for text processing and
        classification tasks. The model must be an instance of YOLOEModel.

        Args:
            vocab (list): Vocabulary list containing tokens or words used by the model for text processing.
            names (list): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of YOLOEModel.

        Examples:
            >>> model = YOLOE("yoloe-v8s.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes, embeddings):
        """
        Set the model's class names and embeddings for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
            embeddings (torch.Tensor): Embeddings corresponding to the classes.
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_classes(classes, embeddings)
        # Verify no background class is present
        assert " " not in classes
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes

    def val(
        self,
        validator=None,
        load_vp=False,
        refer_data=None,
        **kwargs,
    ):
        """
        Validate the model using text or visual prompts.

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
            refer_data (str, optional): Path to the reference data for visual prompts.
            **kwargs (Any): Additional keyword arguments to override default settings.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        custom = {"rect": not load_vp}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict = {},
        refer_image=None,
        predictor=None,
        **kwargs,
    ):
        """
        Run prediction on images, videos, directories, streams, etc.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths,
                directory paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a
                generator as they are computed.
            visual_prompts (dict): Dictionary containing visual prompts for the model. Must include 'bboxes' and
                'cls' keys when non-empty.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable, optional): Custom predictor function. If None, a predictor is automatically
                loaded based on the task.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (List | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = YOLOE("yoloe-v8s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
        self.predictor = (predictor or self._smart_load("predictor"))(
            overrides={
                "task": self.model.task,
                "mode": "predict",
                "save": False,
                "verbose": refer_image is None,
                "batch": 1,
            },
            _callbacks=self.callbacks,
        )

        if len(visual_prompts):
            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list)  # means multiple images
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())

        self.predictor.setup_model(model=self.model)

        if refer_image is None and source is not None:
            dataset = load_inference_source(source)
            if dataset.mode in {"video", "stream"}:
                # NOTE: set the first frame as refer image for videos/streams inference
                refer_image = next(iter(dataset))[1][0]
        if refer_image is not None and len(visual_prompts):
            vpe = self.predictor.get_vpe(refer_image)
            self.model.set_classes(self.model.names, vpe)
            self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
            self.predictor = None  # reset predictor

        return super().predict(source, stream, **kwargs)
