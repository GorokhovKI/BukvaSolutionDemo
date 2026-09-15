from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch

from model_old import SignLanguageTransformer

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@dataclass(frozen=True)
class ModelConfig:
    """Имена моделей. ONNX всегда проверяется первым."""

    onnx_name: str = "best_model_bukva.onnx"
    pth_name: str = "best_model_bukva.pth"
    sequence_length: int = 80
    input_size: int = 126
    num_classes: int = 33


class Runtime(Protocol):
    backend_name: str
    device_description: str

    def predict_logits(self, sequence: np.ndarray) -> np.ndarray:
        ...


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class OnnxRuntimeBackend:
    backend_name = "ONNX Runtime"

    def __init__(self, path: str, config: ModelConfig):
        if ort is None:
            raise RuntimeError(
                "Найден ONNX-файл, но пакет onnxruntime не установлен. "
                "Установите: pip install onnxruntime"
            )

        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [provider for provider in preferred if provider in available]

        if not providers:
            raise RuntimeError(
                "ONNX Runtime установлен, но не найден ни CUDAExecutionProvider, "
                "ни CPUExecutionProvider."
            )

        self.session = ort.InferenceSession(path, providers=providers)
        self.input_info = self.session.get_inputs()[0]
        self.output_info = self.session.get_outputs()[0]
        self.input_name = self.input_info.name
        self.device_description = ", ".join(self.session.get_providers())

        self._validate_model_shape(config)
        logging.info(
            "ONNX Runtime инициализирован: файл=%s, providers=%s, input=%s",
            path,
            self.device_description,
            self.input_info.shape,
        )

    def _validate_model_shape(self, config: ModelConfig) -> None:
        # Обычно форма: [batch, 80, 126]. None или str означает динамическую ось.
        shape = self.input_info.shape
        if len(shape) != 3:
            raise ValueError(
                "ONNX-модель должна принимать 3D tensor [batch, sequence, features], "
                f"но имеет форму input {shape}."
            )

        expected = [None, config.sequence_length, config.input_size]
        for actual, required, dimension_name in zip(
            shape,
            expected,
            ("batch", "sequence_length", "input_size"),
        ):
            if required is None or actual is None or isinstance(actual, str):
                continue
            if actual != required:
                raise ValueError(
                    f"ONNX input dimension '{dimension_name}'={actual}, "
                    f"но приложение ожидает {required}. "
                    "Проверьте preprocessing и конфигурацию модели."
                )

    def predict_logits(self, sequence: np.ndarray) -> np.ndarray:
        batch = np.ascontiguousarray(sequence[np.newaxis, ...], dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: batch})
        logits = np.asarray(outputs[0], dtype=np.float32)

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(
                "Некорректный output ONNX-модели. Ожидаются logits формы [1, classes], "
                f"получено: {logits.shape}."
            )
        return logits


class PyTorchBackend:
    backend_name = "PyTorch"

    def __init__(self, path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_description = str(self.device)
        self.model = SignLanguageTransformer().to(self.device)

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            # Поддержка PyTorch, где аргумент weights_only ещё не существует.
            checkpoint = torch.load(path, map_location=self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logging.info("PyTorch checkpoint инициализирован: файл=%s, device=%s", path, self.device)

    def predict_logits(self, sequence: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
        return logits.detach().cpu().numpy().astype(np.float32, copy=False)


class SignRecognizer:
    """
    Унифицированный runtime для desktop-приложения.

    Порядок выбора строго следующий:
    1. best_model_bukva.onnx;
    2. best_model_bukva.pth.

    Если ONNX существует, но сломан или onnxruntime не установлен,
    приложение намеренно НЕ переключается молча на .pth. Ошибка видна
    в Debug Console, поскольку ONNX задан как приоритетный production-runtime.
    """

    def __init__(self, config: ModelConfig, resource_path):
        self.config = config
        self.backend: Runtime
        self.model_path: str

        onnx_path = resource_path(config.onnx_name)
        pth_path = resource_path(config.pth_name)

        if os.path.isfile(onnx_path):
            print(f"Найдена ONNX-модель (приоритет): {onnx_path}")
            self.backend = OnnxRuntimeBackend(onnx_path, config)
            self.model_path = onnx_path
        elif os.path.isfile(pth_path):
            print(f"ONNX-модель не найдена. Используется PyTorch checkpoint: {pth_path}")
            self.backend = PyTorchBackend(pth_path)
            self.model_path = pth_path
        else:
            raise FileNotFoundError(
                "Не найдены файлы модели. Ожидается один из файлов:\n"
                f"  1) {onnx_path}  (приоритетный ONNX runtime)\n"
                f"  2) {pth_path}   (резервный PyTorch runtime)"
            )

        print(
            f"Backend модели: {self.backend.backend_name}; "
            f"устройство/providers: {self.backend.device_description}"
        )

    @property
    def backend_description(self) -> str:
        return f"{self.backend.backend_name} ({self.backend.device_description})"

    def predict(self, sequence: np.ndarray, index_to_class: dict[int, str]) -> tuple[str, float]:
        expected_shape = (self.config.sequence_length, self.config.input_size)
        if sequence.shape != expected_shape:
            raise ValueError(
                "Некорректная форма входной последовательности: "
                f"{sequence.shape}; ожидается {expected_shape}."
            )

        logits = self.backend.predict_logits(sequence)
        if logits.shape[1] != self.config.num_classes:
            raise ValueError(
                "Число выходных классов модели не совпадает с числом классов приложения: "
                f"модель={logits.shape[1]}, приложение={self.config.num_classes}."
            )

        probabilities = stable_softmax(logits)
        class_index = int(np.argmax(probabilities[0]))
        confidence = float(probabilities[0, class_index])

        return index_to_class[class_index], confidence