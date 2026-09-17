# RSL Subtitles

Desktop-приложение для распознавания букв русского жестового языка (РЖЯ) по видеопотоку с веб-камеры. Программа обнаруживает руки через MediaPipe Hands, формирует временную последовательность landmark-признаков и классифицирует жест с помощью Transformer-модели. Распознанные символы добавляются в строку субтитров.

> Проект является демонстрационным и исследовательским приложением. Он не заменяет профессиональный перевод РЖЯ и не предназначен для критических сценариев без дополнительной валидации модели, UX и качества распознавания.

## Возможности

- Desktop-интерфейс на PySide6 без обязательной внешней консоли.
- Выбор и смена веб-камеры из графического меню.
- Детекция до двух рук с помощью MediaPipe Hands.
- Обработка 21 landmark для каждой руки: `x`, `y`, `z`.
- Фиксированное размещение рук в признаковом векторе: `Left` и `Right`.
- Transformer-классификатор для последовательности из 80 кадров.
- Приоритетная загрузка ONNX-модели через ONNX Runtime.
- Резервная загрузка PyTorch checkpoint (`.pth`), если ONNX-файл отсутствует.
- Сглаживание предсказаний по окну последних inference-результатов.
- Ввод символа только после стабильного удержания жеста.
- Субтитры, пробел, удаление последнего символа и очистка текста.
- Встроенная Debug Console: в неё перенаправляются `print()`, runtime-ошибки и traceback.
- Сборка Windows-приложения через PyInstaller.
- Создание установщика Windows через Inno Setup.

## Демонстрация работы

Рабочий pipeline приложения:

```text
Веб-камера
    ↓
OpenCV VideoCapture
    ↓
MediaPipe Hands
    ↓
Нормализация landmark-координат
    ↓
Буфер временной последовательности: 80 × 126
    ↓
ONNX Runtime (приоритет) / PyTorch (fallback)
    ↓
Softmax и сглаживание классов
    ↓
Проверка уверенности и стабильности жеста
    ↓
Субтитры в desktop-интерфейсе
```

## Поддерживаемые классы

Модель распознаёт 33 буквы русского алфавита:

```text
А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я
```

Каждый выходной индекс модели обязан соответствовать следующему порядку классов:

```python
CLASSES = [
    "А", "Б", "В", "Г", "Д", "Е", "Ё", "Ж", "З", "И", "Й",
    "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф",
    "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я",
]
```

Если порядок классов при обучении отличается от порядка в приложении, предсказания будут неверно интерпретированы.

## Структура проекта

```text
RSL-Subtitles/
├── rsl_subtitles_desktop.py      # Главное desktop-приложение
├── inference_backend.py           # Выбор ONNX/PyTorch runtime и inference
├── model.py                       # Определение SignLanguageTransformer для .pth
├── best_model_bukva.onnx          # ONNX-модель, основной runtime
├── best_model_bukva.pth           # PyTorch checkpoint, резервный runtime
├── LiberationSans-Regular.ttf     # Шрифт для корректной кириллицы
├── requirements.txt                # Зависимости разработки
├── README.md
├── .gitignore
├── screenshots/                    # Пользовательские скриншоты; не коммитятся
└── logs/                           # Runtime-логи при включении записи в файл
```

## Системные требования

### Для запуска из исходного кода

- Windows 10 или Windows 11, 64-bit.
- Python 3.10–3.11, 64-bit.
- Веб-камера.
- Минимум 4 ГБ RAM; рекомендуется 8 ГБ RAM.
- NVIDIA GPU необязательна: ONNX Runtime может работать на CPU.

### Для конечного пользователя

При использовании готового `Setup.exe` отдельная установка Python, PyTorch, MediaPipe, PySide6 или ONNX Runtime не требуется.

## Быстрый запуск

### 1. Клонирование репозитория

```powershell
git clone https://github.com/<YOUR_GITHUB_LOGIN>/RSL-Subtitles.git
cd RSL-Subtitles
```

Если модели не хранятся в репозитории из-за размера, скачайте их из Assets нужного GitHub Release и поместите в корень проекта.

### 2. Создание виртуального окружения

```powershell
py -3.11 -m venv .venv
```

Активация в PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Если политика PowerShell блокирует активацию, можно запускать Python окружения напрямую:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Установка зависимостей

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Добавление модели

Положите в корень проекта один или оба файла:

```text
best_model_bukva.onnx
best_model_bukva.pth
```

Приоритет выбора:

```text
best_model_bukva.onnx  → ONNX Runtime
best_model_bukva.pth   → PyTorch fallback, только если ONNX отсутствует
```

### 5. Запуск

```powershell
python rsl_subtitles_desktop.py
```

При старте откроется GUI-диалог выбора камеры.

## Зависимости

Пример `requirements.txt`:

```text
numpy
opencv-python
mediapipe
Pillow
PySide6
onnxruntime
torch
```

Назначение пакетов:

| Пакет | Назначение |
|---|---|
| `numpy` | Хранение и преобразование массивов landmark-признаков |
| `opencv-python` | Захват кадров с камеры и базовые преобразования изображений |
| `mediapipe` | Обнаружение и трекинг landmark рук |
| `Pillow` | Отрисовка кириллического текста поверх видеокадра |
| `PySide6` | Desktop-интерфейс, меню, диалоги, Debug Console |
| `onnxruntime` | Основной inference backend для `.onnx` |
| `torch` | Резервный inference backend для `.pth` и загрузка архитектуры |

> Для работы ONNX на CPU используйте `onnxruntime`. Для GPU-режима требуется отдельная совместимая установка `onnxruntime-gpu`; сначала рекомендуется протестировать CPU-сборку.

## Управление в приложении

| Действие | Меню | Горячая клавиша |
|---|---|---|
| Выбрать или сменить камеру | `Камера → Выбрать камеру` | `Ctrl + M` |
| Пауза/продолжение inference | `Камера → Пауза` | `P` |
| Сбросить временной буфер | `Камера → Сбросить буфер` | `R` |
| Добавить пробел | `Субтитры → Пробел` | `Space` |
| Удалить последний символ | `Субтитры → Удалить` | `Backspace` |
| Очистить строку субтитров | `Субтитры → Очистить` | `C` |
| Открыть debug-панель | `Отладка → Debug Console` | `Ctrl + D` |

## Debug Console и диагностика

Приложение намеренно содержит встроенную панель диагностики, чтобы ошибки были видны и в сборке с `--windowed`, где внешняя консоль отсутствует.

В Debug Console должны отображаться сообщения, похожие на:

```text
Найдена ONNX-модель (приоритет): ...\best_model_bukva.onnx
Backend модели: ONNX Runtime; устройство/providers: CPUExecutionProvider
Камера 0; runtime: ONNX Runtime (CPUExecutionProvider)
Добавлен символ: А; текст: А
```

При ошибке приложение показывает стандартное окно Windows и выводит полный traceback в Debug Console.

### Типичные проблемы

| Симптом | Возможная причина | Решение |
|---|---|---|
| `No module named PySide6` | Не установлена Qt-библиотека | `python -m pip install PySide6` |
| `No module named PyInstaller` | PyInstaller установлен не в текущем `.venv` | `python -m pip install pyinstaller` |
| Не найден checkpoint | В корне нет `.onnx` и `.pth` | Добавьте `best_model_bukva.onnx` или `best_model_bukva.pth` |
| ONNX найден, но не запускается | Не установлен `onnxruntime` или ONNX имеет неверную форму | Установите `onnxruntime`, проверьте вход/выход модели |
| Нет изображения с камеры | Камера занята браузером/мессенджером или запрещён доступ | Закройте другое ПО и проверьте разрешения камеры Windows |
| Ошибка при смене камеры | Старый поток камеры не завершился корректно | Используйте штатную остановку worker без генерации ошибки при `running=False` |
| Кириллица отображается неправильно | Нет файла шрифта | Добавьте `LiberationSans-Regular.ttf` в корень проекта и в PyInstaller data |

## Контракт данных модели

Качество распознавания зависит от полного совпадения preprocessing во время обучения и во время inference.

### Форма входа

```text
[batch_size, sequence_length, input_size]
[1, 80, 126]
```

Где:

```text
80  = число кадров во временном окне
126 = 2 руки × 21 landmark × 3 координаты (x, y, z)
```

### Нормализация

Для каждой руки выполняются следующие операции:

1. Landmark запястья (`0`) используется как начало координат.
2. Все координаты центрируются относительно запястья.
3. Масштаб вычисляется как расстояние до landmark `9`.
4. Координаты делятся на этот масштаб.
5. Значения ограничиваются диапазоном `[-5.0, 5.0]`.

### Порядок рук

В runtime используется фиксированная схема:

```text
slot 0 → Left hand
slot 1 → Right hand
```

Это критично. Датасет и экспортированная модель должны использовать тот же порядок. Если во время обучения руки записывались просто в порядке обнаружения MediaPipe, рекомендуется привести preprocessing датасета к фиксированной схеме и переобучить модель.

## ONNX и PyTorch runtime

`inference_backend.py` выбирает backend в следующем порядке:

```text
1. best_model_bukva.onnx
2. best_model_bukva.pth
3. FileNotFoundError, если обе модели отсутствуют
```

Если ONNX-файл существует, но повреждён, несовместим или не установлен ONNX Runtime, приложение должно показать ошибку в Debug Console, а не молча переходить на `.pth`. Это помогает обнаружить проблемы релизной ONNX-модели до распространения приложения.

### Проверка ONNX-модели

Ожидается:

```text
Input:  float32 [1, 80, 126] или динамический batch [None, 80, 126]
Output: logits [1, 33]
```

Перед релизом сравните выходы `.pth` и `.onnx` на одинаковых входных последовательностях. Класс с максимальной вероятностью должен совпадать, а различия вероятностей должны быть малыми.

## Экспорт PyTorch в ONNX

Пример отдельного скрипта `export_onnx.py`:

```python
import torch
from model import SignLanguageTransformer

CHECKPOINT_PATH = "best_model_bukva.pth"
ONNX_PATH = "best_model_bukva.onnx"

model = SignLanguageTransformer()
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
model.eval()

dummy_input = torch.randn(1, 80, 126, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["landmarks"],
    output_names=["logits"],
    dynamic_axes={
        "landmarks": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
    do_constant_folding=True,
)

print(f"ONNX-модель сохранена: {ONNX_PATH}")
```

## Сборка portable Windows-версии

Установите PyInstaller:

```powershell
python -m pip install pyinstaller
```

В PowerShell из корневой папки проекта:

```powershell
py -m PyInstaller --noconfirm --clean --onedir --windowed --name RSLSubtitles `
  --add-data "LiberationSans-Regular.ttf;." `
  --add-data "best_model_bukva.onnx;." `
  --add-data "best_model_bukva.pth;." `
  --collect-all mediapipe `
  --collect-all onnxruntime `
  --collect-all PySide6 `
  --hidden-import onnxruntime.capi._pybind_state `
  rsl_subtitles_desktop.py
```

Готовое приложение появится здесь:

```text
dist\RSLSubtitles\RSLSubtitles.exe
```

Перед созданием установщика всегда проверьте запуск этого `.exe` вручную.

> Для `cmd.exe` замените PowerShell-символ продолжения строки `` ` `` на `^`.

## Создание Windows Setup.exe

1. Соберите приложение через PyInstaller.
2. Установите Inno Setup 6.
3. Откройте `RSLSubtitlesInstaller.iss` в Inno Setup Compiler.
4. Проверьте, что `.iss` находится в корне проекта, а не в `dist\RSLSubtitles`.
5. Нажмите `Build → Compile` или `F9`.

Итоговый установщик будет создан в папке:

```text
installer_output\RSLSubtitlesSetup_1.0.0.exe
```

В релиз распространяется именно файл `RSLSubtitlesSetup_<VERSION>.exe`, а не содержимое `dist`.

## Публикация на GitHub

В репозиторий коммитятся исходники, конфигурация сборки, README и документация. В Git обычно не добавляются:

```text
.venv/
build/
dist/
installer_output/
__pycache__/
logs/
screenshots/
```

Готовый `RSLSubtitlesSetup_1.0.0.exe` прикрепляется в GitHub через:

```text
Repository → Releases → Draft a new release → Attach binaries → Publish release
```

Для первого стабильного релиза используйте:

```text
Git tag:       v1.0.0
Release title: RSL Subtitles v1.0.0
Asset:         RSLSubtitlesSetup_1.0.0.exe
```

Если ONNX или `.pth` превышают лимит обычного Git-файла, не добавляйте их в commit: размещайте их в release assets, Git LFS или в отдельном хранилище.

## Тестирование перед релизом

Перед публикацией выполните минимальный checklist:

```text
[ ] Запуск приложения из исходного кода в новом `.venv`
[ ] Загрузка ONNX как приоритетной модели
[ ] Проверка fallback на `.pth` при временном удалении ONNX
[ ] Корректная форма model input: [1, 80, 126]
[ ] Корректное число классов: 33
[ ] Проверка ввода символа после удержания жеста
[ ] Проверка повторного ввода символа после смены жеста/пропадания руки
[ ] Проверка Space, Backspace и Clear
[ ] Проверка нескольких камер и переключения камеры
[ ] Проверка Debug Console и traceback
[ ] Проверка сборки `dist\RSLSubtitles\RSLSubtitles.exe`
[ ] Проверка `Setup.exe` на чистой Windows-машине без Python
[ ] Проверка деинсталляции через Windows Settings
```

## Безопасность и приватность

- Видеокадры обрабатываются локально на устройстве, если в код не добавлены сетевые функции.
- Приложение не должно записывать видео, фото или landmark-данные без явного уведомления пользователя.
- Не публикуйте в репозитории персональные записи жестов, содержащие лица, голоса, имена, пути к локальным файлам или приватные данные.
- Если в будущем добавится аналитика, отправка логов или облачный inference, это необходимо явно документировать и получать согласие пользователя.

## Ограничения

- Точность распознавания определяется качеством и балансом датасета.
- Жесты, похожие по конфигурации пальцев, могут путаться.
- Освещение, фон, угол камеры, перекрытие рук и низкое разрешение снижают качество.
- Текущая модель распознаёт отдельные буквы, а не непрерывную полноценную речь РЖЯ.
- Стабильность результатов зависит от соответствия preprocessing между train и inference.
- Работа GPU через ONNX Runtime зависит от драйверов NVIDIA, CUDA и совместимости версий; CPU backend является базовым переносимым вариантом.

## Roadmap

- [ ] Настройки confidence threshold и времени стабильности из GUI.
- [ ] Сохранение/экспорт текста субтитров в `.txt`.
- [ ] История распознанных символов и событий.
- [ ] Запись диагностических логов в пользовательскую папку.
- [ ] Выбор ONNX-модели через GUI.
- [ ] Проверка совместимости модели до запуска камеры.
- [ ] Автоматические тесты preprocessing, runtime и SubtitleManager.
- [ ] CI/CD для сборки Windows release.
- [ ] Подпись Windows-установщика code-signing сертификатом.
- [ ] Расширение словаря от букв к словам и динамическим жестам.

## Contributing

Перед изменением кода:

1. Создайте отдельную ветку.
2. Не коммитьте `.venv`, `dist`, `build` и пользовательские данные.
3. Сохраняйте совместимость preprocessing с обученной моделью.
4. Проверяйте приложение с ONNX и `.pth`, если оба backend поддерживаются.
5. Добавляйте тесты для исправленных ошибок, если в проекте введён test suite.
6. Описывайте изменения в Pull Request: что изменено, как проверено, есть ли влияние на формат модели.

## Контакты

Для вопросов, bug reports и предложений используйте GitHub Issues этого репозитория. При создании issue приложите:

- версию приложения;
- Windows version;
- тип и индекс камеры;
- используемый backend: ONNX Runtime или PyTorch;
- содержимое Debug Console / traceback;
- шаги для воспроизведения ошибки;
- ожидаемое и фактическое поведение.
