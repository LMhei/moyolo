# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import inspect
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.hub import HUB_WEB_ROOT, HUBTrainingSession
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)


class Model(torch.nn.Module):
    """
    实现YOLO模型的基类，统一不同模型类型的API接口。

    该类为YOLO模型相关操作提供统一接口，包括训练、验证、预测、导出和基准测试等。
    支持处理多种模型类型，包括从本地文件、Ultralytics HUB或Triton Server加载的模型。

    属性:
        callbacks (dict): 用于模型操作期间各类事件回调函数的字典
        predictor (BasePredictor): 用于执行预测的预测器对象
        model (torch.nn.Module): 底层的PyTorch模型
        trainer (BaseTrainer): 用于模型训练的训练器对象
        ckpt (dict): 如果从*.pt文件加载模型，包含的检查点数据
        cfg (str): 如果从*.yaml文件加载模型，包含的模型配置
        ckpt_path (str): 检查点文件路径
        overrides (dict): 模型配置覆盖参数的字典
        metrics (dict): 最新的训练/验证指标
        session (HUBTrainingSession): Ultralytics HUB会话对象（如果适用）
        task (str): 模型的任务类型（如检测/分割/分类等）
        model_name (str): 模型名称

    方法:
        __call__: predict方法的别名，使模型实例可被直接调用
        _new: 根据配置文件初始化新模型
        _load: 从检查点文件加载模型
        _check_is_pytorch_model: 确保模型是PyTorch模型
        reset_weights: 将模型权重重置为初始状态
        load: 从指定文件加载模型权重
        save: 将当前模型状态保存到文件
        info: 记录或返回模型信息
        fuse: 融合Conv2d和BatchNorm2d层以优化推理
        predict: 执行目标检测预测
        track: 执行目标跟踪
        val: 在数据集上验证模型
        benchmark: 对不同导出格式进行性能基准测试
        export: 将模型导出为不同格式
        train: 在数据集上训练模型
        tune: 执行超参数调优
        _apply: 对模型张量应用指定函数
        add_callback: 为事件添加回调函数
        clear_callback: 清除事件的所有回调函数
        reset_callbacks: 重置所有回调函数为默认实现

    示例:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")  # 加载预训练模型
        >>> results = model.predict("image.jpg")  # 执行图像预测
        >>> model.train(data="coco8.yaml", epochs=3)  # 进行3轮训练
        >>> metrics = model.val()  # 验证模型性能
        >>> model.export(format="onnx")  # 导出为ONNX格式
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        """
        初始化YOLO模型类的新实例

        该构造函数根据提供的模型路径或名称设置模型。支持处理多种模型来源，
        包括本地文件、Ultralytics HUB云端模型和Triton Server服务模型。
        该方法初始化模型的多个重要属性，为后续训练、预测或导出等操作做好准备

        Args:
            model (str | Path): 要加载或创建的模型路径/名称。可以是：
                - 本地文件路径（*.pt/*.yaml）
                - Ultralytics HUB模型名称
                - Triton Server服务模型名称
            task (str | None): 模型关联的任务类型，指定其应用领域（如检测/分割/分类）
            verbose (bool): 详细模式，若为True则在初始化和后续操作中显示详细输出

        Raises:
            FileNotFoundError: 当指定模型文件不存在或无法访问时抛出
            ValueError: 当模型文件/配置无效或不支持时抛出
            ImportError: 当特定模型类型（如HUB SDK）所需依赖未安装时抛出

        Examples:
            >>> model = Model("yolo11n.pt")  # 加载预训练权重
            >>> model = Model("path/to/model.yaml", task="detect")  # 从配置文件初始化检测模型
            >>> model = Model("hub_model", verbose=True)  # 从HUB加载模型并启用详细日志
        """
        super().__init__()
        # 初始化默认回调函数集合（训练/验证/预测等阶段的钩子函数）
        self.callbacks = callbacks.get_default_callbacks()

        # 预测器对象（推理时复用）
        self.predictor = None  # reuse predictor

        # 存储底层PyTorch模型对象（如YOLOv8/YOLO-World等网络结构）
        self.model = None  # model object

        # 训练器对象（包含优化器/学习率调度器等训练组件）
        self.trainer = None  # trainer object

        # 检查点字典（当从*.pt文件加载时存储权重/超参数等信息）
        self.ckpt = {}  # if loaded from *.pt

        # 模型配置信息（当从*.yaml文件加载时存储网络结构配置）
        self.cfg = None  # if loaded from *.yaml

        # 检查点文件路径（如：'yolov8n.pt'）
        self.ckpt_path = None

        # 训练参数覆盖字典（用于动态修改训练配置）
        self.overrides = {}  # overrides for trainer object

        # 评估指标字典（存储mAP50/mAP50-95等训练验证指标）
        self.metrics = None  # validation/training metrics

        # Ultralytics HUB会话对象（云端训练/部署时使用）
        self.session = None  # HUB session

        # 任务类型标识（如'detect'/'segment'/'classify'等）
        self.task = task  # task type

        # 模型名称标识（如'yolov8n'/'yolo-world-l'等）
        self.model_name = None  # model name
        model = str(model).strip()

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):
            # Fetch model from HUB
            checks.check_requirements("hub-sdk>=0.0.12")
            session = HUBTrainingSession.create_session(model)
            model = session.model_file
            if session.train_args:  # training sent from HUB
                self.session = session

        # Check if Triton Server model
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"  # set `task=detect` if not explicitly set
            return

        # Load or create new YOLO model
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to avoid deterministic warnings
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

        # Delete super().training for accessing self.model.training
        del self.training

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        预测方法的别名，使模型实例可被直接调用进行预测

该方法通过允许直接调用模型实例简化预测流程，支持传入多种输入格式进行预测

Args:
    source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple):
        预测图像来源，支持以下格式：
        - 文件路径/URL
        - PIL图像对象
        - numpy数组 (HWC格式)
        - PyTorch张量
        - 摄像头设备ID (int)
        - 上述类型的列表/元组组合
    stream (bool): 流模式，若为True则将输入源视为连续视频流进行预测
    **kwargs (Any): 其他预测配置参数（置信度/IOU阈值等）

Returns:
    (List[ultralytics.engine.results.Results]): 预测结果列表，每个元素为Results对象

Examples:
    >>> model = YOLO("yolo11n.pt")
    >>> results = model("https://ultralytics.com/images/bus.jpg")  # 对单张图片进行预测
    >>> for r in results:
    ...     print(f"检测到图像中有 {len(r)} 个物体")  # 输出：检测到图像中有 4 个物体
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """
        检查模型字符串是否为Triton Server服务地址

        该静态方法通过urllib.parse.urlsplit()解析模型字符串组成部分，
        判断是否为有效的Triton Server服务地址

        Args:
            model (str): 待检查的模型字符串

        Returns:
            (bool): 如果是有效的Triton Server地址返回True，否则返回False

        Examples:
            >>> Model.is_triton_model("http://localhost:8000/v2/models/yolo11n")  # Triton服务地址
            True
            >>> Model.is_triton_model("yolo11n.pt")  # 本地模型文件路径
            False
        """
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """
        检查模型是否为Ultralytics HUB云端模型

        该静态方法用于判断给定模型字符串是否为有效的Ultralytics HUB模型标识

        Args:
            model (str): 待检查的模型字符串

        Returns:
            (bool): 如果是有效的HUB模型标识返回True，否则返回False

        Examples:
            >>> Model.is_hub_model("https://hub.ultralytics.com/models/MODEL")  # HUB模型地址
            True
            >>> Model.is_hub_model("yolo11n.pt")  # 本地模型文件路径
            False
        """
        return model.startswith(f"{HUB_WEB_ROOT}/models/")

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        初始化新模型并从配置定义推断任务类型

        根据提供的配置文件创建新的模型实例。加载模型配置，
        若未明确指定任务类型则自动推断，并通过任务映射表初始化对应类型的模型类

        Args:
            cfg (str): 模型配置文件路径（YAML格式）
            task (str | None): 指定模型任务类型。若为None则从配置自动推断
            model (torch.nn.Module | None): 自定义PyTorch模型实例，若提供则直接使用而不新建
            verbose (bool): 详细模式，若为True则在加载时显示模型信息

        Raises:
            ValueError: 当配置文件无效或无法推断任务类型时抛出
            ImportError: 当指定任务所需依赖未安装时抛出

        Examples:
            >>> model = Model()
            >>> model._new("yolo11n.yaml", task="detect", verbose=True)  # 从配置文件创建检测模型并启用详细输出
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """
        Load a model from a checkpoint file or initialize it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
        up the model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str | None): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.

        Examples:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolo11n -> yolo11n.pt

        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        """
        Check if the model is a PyTorch model and raise TypeError if it's not.

        This method verifies that the model is either a PyTorch module or a .pt file. It's used to ensure that
        certain operations that require a PyTorch model are only performed on compatible model types.

        Raises:
            TypeError: If the model is not a PyTorch module or a .pt file. The error message provides detailed
                information about supported model formats and operations.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model._check_is_pytorch_model()  # No error raised
            >>> model = Model("yolo11n.onnx")
            >>> model._check_is_pytorch_model()  # Raises TypeError
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, torch.nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolo11n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "Model":
        """
        Reset the model's weights to their initial state.

        This method iterates through all modules in the model and resets their parameters if they have a
        'reset_parameters' method. It also ensures that all parameters have 'requires_grad' set to True,
        enabling them to be updated during training.

        Returns:
            (Model): The instance of the class with reset weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.reset_weights()
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolo11n.pt") -> "Model":
        """
        Load parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (Union[str, Path]): Path to the weights file or a weights object.

        Returns:
            (Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model()
            >>> model.load("yolo11n.pt")
            >>> model.load(Path("path/to/weights.pt"))
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights  # remember the weights for DDP training
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt") -> None:
        """
        Save the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename. It includes metadata such as
        the date, Ultralytics version, license information, and a link to the documentation.

        Args:
            filename (str | Path): The name of the file to save the model to.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.save("my_model.pt")
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Display model information.

        This method provides an overview or detailed information about the model, depending on the arguments
        passed. It can control the verbosity of the output and return the information as a list.

        Args:
            detailed (bool): If True, shows detailed information about the model layers and parameters.
            verbose (bool): If True, prints the information. If False, returns the information as a list.

        Returns:
            (List[str]): A list of strings containing various types of information about the model, including
                model summary, layer details, and parameter counts. Empty if verbose is True.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.info()  # Prints model summary
            >>> info_list = model.info(detailed=True, verbose=False)  # Returns detailed info as a list
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self) -> None:
        """
        Fuse Conv2d and BatchNorm2d layers in the model for optimized inference.

        This method iterates through the model's modules and fuses consecutive Conv2d and BatchNorm2d layers
        into a single layer. This fusion can significantly improve inference speed by reducing the number of
        operations and memory accesses required during forward passes.

        The fusion process typically involves folding the BatchNorm2d parameters (mean, variance, weight, and
        bias) into the preceding Conv2d layer's weights and biases. This results in a single Conv2d layer that
        performs both convolution and normalization in one step.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.fuse()
            >>> # Model is now fused and ready for optimized inference
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        Generate image embeddings based on the provided source.

        This method is a wrapper around the 'predict()' method, focusing on generating embeddings from an image
        source. It allows customization of the embedding process through various keyword arguments.

        Args:
            source (str | Path | int | List | Tuple | np.ndarray | torch.Tensor): The source of the image for
                generating embeddings. Can be a file path, URL, PIL image, numpy array, etc.
            stream (bool): If True, predictions are streamed.
            **kwargs (Any): Additional keyword arguments for configuring the embedding process.

        Returns:
            (List[torch.Tensor]): A list containing the image embeddings.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> print(embeddings[0].shape)
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> List[Results]:
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): The source
                of the image(s) to make predictions on. Accepts various types including file paths, URLs, PIL
                images, numpy arrays, and torch tensors.
            stream (bool): If True, treats the input source as a continuous stream for predictions.
            predictor (BasePredictor | None): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor.
            **kwargs (Any): Additional keyword arguments for configuring the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # print detection bounding boxes

        Notes:
            - If 'source' is not provided, it defaults to the ASSETS constant with a warning.
            - The method sets up a new predictor if not already present and updates its arguments with each call.
            - For SAM-type models, 'prompts' can be passed as a keyword argument.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict", "rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> List[Results]:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It handles
        various input sources such as file paths or video streams, and supports customization through keyword arguments.
        The method registers trackers if not already present and can persist them between calls.

        Args:
            source (Union[str, Path, int, List, Tuple, np.ndarray, torch.Tensor], optional): Input source for object
                tracking. Can be a file path, URL, or video stream.
            stream (bool): If True, treats the input source as a continuous video stream.
            persist (bool): If True, persists trackers between different calls to this method.
            **kwargs (Any): Additional keyword arguments for configuring the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, each a Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # print tracking IDs

        Notes:
            - This method sets a default confidence threshold of 0.1 for ByteTrack-based tracking.
            - The tracking mode is explicitly set in the keyword arguments.
            - Batch size is set to 1 for tracking in videos.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """
        Validate the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for customization through various settings. It
        supports validation with a custom validator or the default validation approach. The method combines default
        configurations, method-specific defaults, and user-provided arguments to configure the validation process.

        Args:
            validator (ultralytics.engine.validator.BaseValidator | None): An instance of a custom validator class for
                validating the model.
            **kwargs (Any): Arbitrary keyword arguments for customizing the validation process.

        Returns:
            (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # Print mAP50-95
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs: Any,
    ):
        """
        Benchmark the model across various export formats to evaluate performance.

        This method assesses the model's performance in different export formats, such as ONNX, TorchScript, etc.
        It uses the 'benchmark' function from the ultralytics.utils.benchmarks module. The benchmarking is
        configured using a combination of default configuration values, model-specific arguments, method-specific
        defaults, and any additional user-provided keyword arguments.

        Args:
            **kwargs (Any): Arbitrary keyword arguments to customize the benchmarking process. Common options include:
                - data (str): Path to the dataset for benchmarking.
                - imgsz (int | List[int]): Image size for benchmarking.
                - half (bool): Whether to use half-precision (FP16) mode.
                - int8 (bool): Whether to use int8 precision mode.
                - device (str): Device to run the benchmark on (e.g., 'cpu', 'cuda').
                - verbose (bool): Whether to print detailed benchmark information.
                - format (str): Export format name for specific benchmarking.

        Returns:
            (dict): A dictionary containing the results of the benchmarking process, including metrics for
                different export formats.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose", False),
            format=kwargs.get("format", ""),
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """
        Export the model to a different format suitable for deployment.

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided.

        Args:
            **kwargs (Any): Arbitrary keyword arguments to customize the export process. These are combined with
                the model's overrides and method defaults. Common arguments include:
                format (str): Export format (e.g., 'onnx', 'engine', 'coreml').
                half (bool): Export model in half-precision.
                int8 (bool): Export model in int8 precision.
                device (str): Device to run the export on.
                workspace (int): Maximum memory workspace size for TensorRT engines.
                nms (bool): Add Non-Maximum Suppression (NMS) module to model.
                simplify (bool): Simplify ONNX model.

        Returns:
            (str): The path to the exported model file.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            ValueError: If an unsupported export format is specified.
            RuntimeError: If the export process fails due to errors.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # reset to avoid multi-GPU errors
            "verbose": False,
        }  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        """
        Trains the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings. It supports training with a
        custom trainer or the default training approach. The method handles scenarios such as resuming training
        from a checkpoint, integrating with Ultralytics HUB, and updating model and configuration after training.

        When using Ultralytics HUB, if the session has a loaded model, the method prioritizes HUB training
        arguments and warns if local arguments are provided. It checks for pip updates and combines default
        configurations, method-specific defaults, and user-provided arguments to configure the training process.

        Args:
            trainer (BaseTrainer | None): Custom trainer instance for model training. If None, uses default.
            **kwargs (Any): Arbitrary keyword arguments for training configuration. Common options include:
                data (str): Path to dataset configuration file.
                epochs (int): Number of training epochs.
                batch_size (int): Batch size for training.
                imgsz (int): Input image size.
                device (str): Device to run training on (e.g., 'cuda', 'cpu').
                workers (int): Number of worker threads for data loading.
                optimizer (str): Optimizer to use for training.
                lr0 (float): Initial learning rate.
                patience (int): Epochs to wait for no observable improvement for early stopping of training.

        Returns:
            (Dict | None): Training metrics if available and training is successful; otherwise, None.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)
        """
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Conducts hyperparameter tuning for the model, with an option to use Ray Tune.

        This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method.
        When Ray Tune is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module.
        Otherwise, it uses the internal 'Tuner' class for tuning. The method combines default, overridden, and
        custom arguments to configure the tuning process.

        Args:
            use_ray (bool): Whether to use Ray Tune for hyperparameter tuning. If False, uses internal tuning method.
            iterations (int): Number of tuning iterations to perform.
            *args (Any): Additional positional arguments to pass to the tuner.
            **kwargs (Any): Additional keyword arguments for tuning configuration. These are combined with model
                overrides and defaults to configure the tuning process.

        Returns:
            (dict): Results of the hyperparameter search, including best parameters and performance metrics.

        Raises:
            TypeError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.tune(data="coco8.yaml", iterations=5)
            >>> print(results)

            # Use Ray Tune for more advanced hyperparameter search
            >>> results = model.tune(use_ray=True, iterations=20, data="coco8.yaml")
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> "Model":
        """
        Apply a function to model tensors that are not parameters or registered buffers.

        This method extends the functionality of the parent class's _apply method by additionally resetting the
        predictor and updating the device in the model's overrides. It's typically used for operations like
        moving the model to a different device or changing its precision.

        Args:
            fn (Callable): A function to be applied to the model's tensors. This is typically a method like
                to(), cpu(), cuda(), half(), or float().

        Returns:
            (Model): The model instance with the function applied and updated attributes.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # Move model to GPU
        """
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> Dict[int, str]:
        """
        Retrieves the class names associated with the loaded model.

        This property returns the class names if they are defined in the model. It checks the class names for validity
        using the 'check_class_names' function from the ultralytics.nn.autobackend module. If the predictor is not
        initialized, it sets it up before retrieving the names.

        Returns:
            (Dict[int, str]): A dictionary of class names associated with the model, where keys are class indices and
                values are the corresponding class names.

        Raises:
            AttributeError: If the model or predictor does not have a 'names' attribute.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # export formats will not have predictor defined until predict() is called
            self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        Get the device on which the model's parameters are allocated.

        This property determines the device (CPU or GPU) where the model's parameters are currently stored. It is
        applicable only to models that are instances of torch.nn.Module.

        Returns:
            (torch.device): The device (CPU/GPU) of the model.

        Raises:
            AttributeError: If the model is not a torch.nn.Module instance.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # if CUDA is available
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None

    @property
    def transforms(self):
        """
        Retrieves the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model. The transforms
        typically include preprocessing steps like resizing, normalization, and data augmentation
        that are applied to input data before it is fed into the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"Model transforms: {transforms}")
            ... else:
            ...     print("No transforms defined for this model.")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        Add a callback function for a specified event.

        This method allows registering custom callback functions that are triggered on specific events during
        model operations such as training or inference. Callbacks provide a way to extend and customize the
        behavior of the model at various stages of its lifecycle.

        Args:
            event (str): The name of the event to attach the callback to. Must be a valid event name recognized
                by the Ultralytics framework.
            func (Callable): The callback function to be registered. This function will be called when the
                specified event occurs.

        Raises:
            ValueError: If the event name is not recognized or is invalid.

        Examples:
            >>> def on_train_start(trainer):
            ...     print("Training is starting!")
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        Clears all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event.
        It resets the callback list for the specified event to an empty list, effectively removing all
        registered callbacks for that event.

        Args:
            event (str): The name of the event for which to clear the callbacks. This should be a valid event name
                recognized by the Ultralytics callback system.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", lambda: print("Training started"))
            >>> model.clear_callback("on_train_start")
            >>> # All callbacks for 'on_train_start' are now removed

        Notes:
            - This method affects both custom callbacks added by the user and default callbacks
              provided by the Ultralytics framework.
            - After calling this method, no callbacks will be executed for the specified event
              until new ones are added.
            - Use with caution as it removes all callbacks, including essential ones that might
              be required for proper functioning of certain operations.
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        Reset all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        previously added. It iterates through all default callback events and replaces the current callbacks with the
        default ones.

        The default callbacks are defined in the 'callbacks.default_callbacks' dictionary, which contains predefined
        functions for various events in the model's lifecycle, such as on_train_start, on_epoch_end, etc.

        This method is useful when you want to revert to the original set of callbacks after making custom
        modifications, ensuring consistent behavior across different runs or experiments.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # All callbacks are now reset to their default functions
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        Reset specific arguments when loading a PyTorch model checkpoint.

        This method filters the input arguments dictionary to retain only a specific set of keys that are
        considered important for model loading. It's used to ensure that only relevant arguments are preserved
        when loading a model from a checkpoint, discarding any unnecessary or potentially conflicting settings.

        Args:
            args (dict): A dictionary containing various model arguments and settings.

        Returns:
            (dict): A new dictionary containing only the specified include keys from the input arguments.

        Examples:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key: str):
        """
        智能加载与模型任务匹配的功能模块

        该方法根据模型当前任务类型和提供的模块类型键值(key)，
        动态选择并返回对应的模块类（模型类/训练器类/验证器类/预测器类）。
        通过任务映射表(task_map)字典确定当前任务应加载的特定模块

        Args:
            key (str): 需加载的模块类型，可选值：'model'(模型结构)、'trainer'(训练器)、
                     'validator'(验证器)、'predictor'(预测器)

        Returns:
            (object): 对应当前任务和模块类型的模块类对象

        Raises:
            NotImplementedError: 当当前任务不支持指定的模块类型时抛出

        Examples:
            >>> model = Model(task="detect")  # 创建检测任务模型
            >>> predictor_class = model._smart_load("predictor")  # 获取检测预测器类
            >>> trainer_class = model._smart_load("trainer")  # 获取检测训练器类
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self) -> dict:
        """
        提供从模型任务到不同运行模式对应类的映射关系

        该属性方法返回一个字典，将每个支持的任务（如检测(detect)/分割(segment)/分类(classify)等）
        映射到嵌套字典。嵌套字典包含不同运行模式（模型结构/训练器/验证器/预测器）到各自类实现的映射关系

        通过这种映射机制，可以根据模型任务和运行模式动态加载对应的类实现。
        这种设计使得Ultralytics框架能够灵活扩展，支持多种任务类型的处理

        Returns:
            (Dict[str, Dict[str, Any]]): 字典结构，键为任务名称，值为包含各模式类实现的嵌套字典。
            每个嵌套字典包含以下键：
                - 'model' → 模型结构类（如DetectionModel）
                - 'trainer' → 训练器类（如DetectionTrainer）
                - 'validator' → 验证器类（如DetectionValidator）
                - 'predictor' → 预测器类（如DetectionPredictor）

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> task_map = model.task_map  # 获取任务映射表
            >>> detect_predictor = task_map["detect"]["predictor"]  # 获取检测任务的预测器类
            >>> segment_trainer = task_map["segment"]["trainer"]  # 获取分割任务的训练器类
        """
        raise NotImplementedError("Please provide task map for your model!")

    def eval(self):
        """
        Sets the model to evaluation mode.

        This method changes the model's mode to evaluation, which affects layers like dropout and batch normalization
        that behave differently during training and evaluation. In evaluation mode, these layers use running statistics
        rather than computing batch statistics, and dropout layers are disabled.

        Returns:
            (Model): The model instance with evaluation mode set.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.eval()
            >>> # Model is now in evaluation mode for inference
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """
        Enable accessing model attributes directly through the Model class.

        This method provides a way to access attributes of the underlying model directly through the Model class
        instance. It first checks if the requested attribute is 'model', in which case it returns the model from
        the module dictionary. Otherwise, it delegates the attribute lookup to the underlying model.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            (Any): The requested attribute value.

        Raises:
            AttributeError: If the requested attribute does not exist in the model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.stride)  # Access model.stride attribute
            >>> print(model.names)  # Access model.names attribute
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)
