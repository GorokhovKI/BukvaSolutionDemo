from __future__ import annotations
import os, sys, time, traceback, logging
from collections import deque, Counter
from dataclasses import dataclass
from pathlib import Path
import cv2, mediapipe as mp, numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot, QTimer
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QDialog, QComboBox, QVBoxLayout, QDialogButtonBox, QPushButton, QDockWidget, QPlainTextEdit, QMessageBox, QFileDialog
from model import ModelConfig, SignRecognizer

@dataclass(frozen=True)
class Config:
    onnx_name: str = 'best_model_bukva.onnx'
    pth_name: str = 'best_model_bukva.pth'
    font_name: str = 'LiberationSans-Regular.ttf'
    sequence_length: int = 80
    num_hands: int = 2
    num_landmarks: int = 21
    confidence_threshold: float = .70
    stability_seconds: float = 1.2
    smoothing_window: int = 10
    infer_every: int = 2
    width: int = 1280
    height: int = 720
    @property
    def input_size(self): return self.num_hands * self.num_landmarks * 3

CFG = Config()
CLASSES = ['А','Б','В','Г','Д','Е','Ё','Ж','З','И','Й','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я']
IDX_TO_CLASS = dict(enumerate(CLASSES))

def resource_path(name: str) -> str:
    return os.path.join(getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))), name)

def font(size):
    for p in (resource_path(CFG.font_name), 'arial.ttf', 'DejaVuSans.ttf'):
        try: return ImageFont.truetype(p, size)
        except OSError: pass
    return ImageFont.load_default()

def normalize(landmarks):
    a = np.array([[x.x, x.y, x.z] for x in landmarks], np.float32)
    a -= a[0]; scale = np.linalg.norm(a[9])
    return np.clip(a / max(scale, 1e-6), -5, 5).flatten()

class Bus(QObject):
    message = Signal(str)
class PrintToDebug:
    def __init__(self, bus, level): self.bus, self.level, self.buf = bus, level, ''
    def write(self, value):
        self.buf += value
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            if line.strip(): self.bus.message.emit(f'[{time.strftime("%H:%M:%S")}] [{self.level}] {line}')
        return len(value)
    def flush(self): pass

class Subtitles:
    def __init__(self): self.text=''; self.candidate=None; self.since=None; self.committed=None
    def reset(self): self.candidate=self.since=self.committed=None
    def update(self, char, conf, visible, now):
        if not visible or conf < CFG.confidence_threshold: self.candidate=self.since=None; return
        if char != self.candidate: self.candidate, self.since = char, now; return
        if char != self.committed and self.since and now-self.since >= CFG.stability_seconds:
            self.text += char; self.committed=char; print(f'Добавлен символ: {char}; текст: {self.text}')
    def space(self):
        if self.text and not self.text.endswith(' '): self.text += ' '
    def backspace(self): self.text=self.text[:-1]
    def clear(self): self.text=''; self.reset(); print('Субтитры очищены')

class Worker(QObject):
    frame = Signal(QImage); status = Signal(str); error = Signal(str, str); finished = Signal()
    def __init__(self, camera):
        super().__init__(); self.camera=camera; self.running=True; self.paused=False; self.reset_pending=False
        self.seq=deque(maxlen=CFG.sequence_length); self.history=deque(maxlen=CFG.smoothing_window); self.sub=Subtitles()
        self.pred='...'; self.conf=0.; self.tick=0; self.last_hand=time.monotonic(); self.fps=0.; self.fps_n=0; self.fps_t=time.monotonic()
        self.ui_font, self.sub_font = font(27), font(42)
    def reset(self):
        self.seq.clear(); self.history.clear(); self.sub.reset(); self.pred='...'; self.conf=0.; print('Буфер распознавания сброшен')
    @Slot()
    def run(self):
        cap=hands=None
        try:
            backend=cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY
            cap=cv2.VideoCapture(self.camera, backend); cap.set(cv2.CAP_PROP_FRAME_WIDTH,CFG.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CFG.height); cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            if not cap.isOpened(): raise RuntimeError(f'Не удалось открыть камеру {self.camera}')
            hands=mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=.7, min_tracking_confidence=.5)
            rec=SignRecognizer(ModelConfig(CFG.onnx_name,CFG.pth_name,CFG.sequence_length,CFG.input_size,len(CLASSES)), resource_path)
            print(f'Камера {self.camera}; runtime: {rec.backend_description}')
            while self.running:
                ok, f=cap.read()
                if not ok: raise RuntimeError('Камера перестала передавать кадры')
                f=cv2.flip(f,1); now=time.monotonic()
                if self.reset_pending: self.reset(); self.reset_pending=False
                visible=False
                if not self.paused:
                    rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB); r=hands.process(rgb)
                    features=np.zeros((2,21,3),np.float32)
                    if r.multi_hand_landmarks:
                        visible=True; self.last_hand=now
                        for lm,hd in zip(r.multi_hand_landmarks,r.multi_handedness):
                            slot=0 if hd.classification[0].label=='Left' else 1
                            features[slot]=normalize(lm.landmark).reshape(21,3)
                            mp.solutions.drawing_utils.draw_landmarks(f,lm,mp.solutions.hands.HAND_CONNECTIONS)
                        self.seq.append(features.flatten())
                    elif now-self.last_hand>1.2: self.reset()
                    self.tick+=1
                    if len(self.seq)==CFG.sequence_length and self.tick%CFG.infer_every==0:
                        raw, score=rec.predict(np.asarray(self.seq,np.float32),IDX_TO_CLASS); self.history.append((raw,score))
                        self.pred=Counter(x[0] for x in self.history).most_common(1)[0][0]
                        self.conf=float(np.mean([x[1] for x in self.history if x[0]==self.pred]))
                        self.sub.update(self.pred,self.conf,visible,now)
                self.fps_n+=1
                if now-self.fps_t>=1: self.fps=self.fps_n/(now-self.fps_t); self.fps_n=0; self.fps_t=now
                self.frame.emit(self.render(f)); self.status.emit(f'Камера {self.camera} | {rec.backend_description} | FPS {self.fps:.1f} | {self.pred} {self.conf:.0%}')
        except Exception as e:
            tb=traceback.format_exc(); print('КРИТИЧЕСКАЯ ОШИБКА:',e); print(tb); self.error.emit(str(e),tb)
        finally:
            if hands: hands.close()
            if cap: cap.release()
            self.finished.emit()
    def render(self,f):
        im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im); w,h=im.size
        d.rectangle((0,0,w,95),fill=(0,0,0)); col=(60,255,100) if self.conf>=CFG.confidence_threshold else (255,220,80)
        d.text((15,10),f'Жест: {self.pred} ({self.conf:.0%})',font=self.ui_font,fill=col); d.text((15,50),f'FPS: {self.fps:.1f} | Буфер: {len(self.seq)}/{CFG.sequence_length}',font=self.ui_font,fill='white')
        d.rectangle((0,h-105,w,h),fill=(0,0,0)); txt=self.sub.text or 'Покажите жест и удерживайте его для ввода'
        while d.textbbox((0,0),txt,font=self.sub_font)[2]>w-40: txt=txt[1:]
        d.text((20,h-85),txt,font=self.sub_font,fill='white')
        a = np.ascontiguousarray(np.asarray(im), dtype=np.uint8)
        return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    @Slot()
    def stop(self): self.running=False
    @Slot()
    def pause(self): self.paused=not self.paused; print('Пауза' if self.paused else 'Распознавание продолжено')
    @Slot()
    def request_reset(self): self.reset_pending=True
    @Slot()
    def space(self): self.sub.space()
    @Slot()
    def backspace(self): self.sub.backspace()
    @Slot()
    def clear(self): self.sub.clear()

class CameraDialog(QDialog):
    def __init__(self,parent):
        super().__init__(parent); self.setWindowTitle('Выбор камеры'); self.box=QComboBox(); lay=QVBoxLayout(self); lay.addWidget(QLabel('Выберите камеру:')); lay.addWidget(self.box)
        b=QPushButton('Обновить'); lay.addWidget(b); buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel); lay.addWidget(buttons); b.clicked.connect(self.scan); buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); self.scan()
    def scan(self):
        self.box.clear()
        for i in range(10):
            c=cv2.VideoCapture(i,cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY); ok=c.isOpened() and c.read()[0]; c.release()
            if ok:self.box.addItem(f'Камера {i}',i)
        if not self.box.count(): self.box.addItem('Камеры не найдены',None)
    def camera(self): return self.box.currentData()

class Window(QMainWindow):
    def __init__(self,bus):
        super().__init__(); self.bus=bus; bus.message.connect(self.log); self.thread=None; self.worker=None; self.image=None
        self.setWindowTitle('RSL Subtitles Desktop'); self.resize(1280,820); self.video=QLabel('Выберите камеру: Камера → Выбрать камеру'); self.video.setAlignment(Qt.AlignCenter); self.video.setStyleSheet('background:#181818;color:white;font-size:18px'); self.setCentralWidget(self.video)
        self.debug=QPlainTextEdit(); self.debug.setReadOnly(True); dock=QDockWidget('Debug Console',self); dock.setWidget(self.debug); self.addDockWidget(Qt.BottomDockWidgetArea,dock); self.dock=dock; dock.hide()
        self.make_action('Выбрать камеру…','Ctrl+M',self.choose,'Камера'); self.make_action('Пауза','P',self.call('pause'),'Камера'); self.make_action('Сбросить буфер','R',self.call('request_reset'),'Камера'); self.make_action('Пробел','Space',self.call('space'),'Субтитры'); self.make_action('Удалить','Backspace',self.call('backspace'),'Субтитры'); self.make_action('Очистить','C',self.call('clear'),'Субтитры')
        m=self.menuBar().addMenu('Отладка'); a=QAction('Debug Console',self); a.setShortcut('Ctrl+D'); a.triggered.connect(lambda:self.dock.setVisible(not self.dock.isVisible())); m.addAction(a)
    def make_action(self,name,key,fn,menu):
        a=QAction(name,self); a.setShortcut(QKeySequence(key)); a.triggered.connect(fn); self.menuBar().findChild(type(self.menuBar())) if False else None; getattr(self,'_menus',set())
        target=next((x for x in self.menuBar().actions() if x.text()==menu),None)
        (target.menu() if target else self.menuBar().addMenu(menu)).addAction(a)
    def call(self,name):
        return lambda: getattr(self.worker,name)() if self.worker else None
    def log(self,s): self.debug.appendPlainText(s)
    def choose(self):
        d=CameraDialog(self)
        if d.exec() and d.camera() is not None: self.start(int(d.camera()))
    def start(self,cam):
        self.stop(); self.thread=QThread(self); self.worker=Worker(cam); self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run); self.worker.frame.connect(self.show_frame); self.worker.status.connect(self.statusBar().showMessage); self.worker.error.connect(self.fail); self.worker.finished.connect(self.thread.quit); self.thread.start()
    def stop(self):
        if self.worker: self.worker.stop()
        if self.thread and self.thread.isRunning(): self.thread.wait(3000)
        self.worker=self.thread=None
    def show_frame(self,img): self.image=img; self.video.setPixmap(QPixmap.fromImage(img).scaled(self.video.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def fail(self,e,t): self.dock.show(); self.log(t); QMessageBox.critical(self,'Ошибка распознавания',e+'\n\nПодробности доступны в Debug Console.')
    def closeEvent(self,e): self.stop(); e.accept()

def main():
    app=QApplication(sys.argv); bus=Bus(); sys.stdout=PrintToDebug(bus,'PRINT'); sys.stderr=PrintToDebug(bus,'ERROR'); w=Window(bus); w.show(); QTimer.singleShot(0,w.choose); return app.exec()
if __name__=='__main__': raise SystemExit(main())
