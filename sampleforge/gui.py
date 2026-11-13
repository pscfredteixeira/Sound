"""PySide6 desktop UI. Audio inference runs in a killable child process."""
from __future__ import annotations

import json
import sys
import threading
import uuid
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSlider, QSplitter,
                               QTextEdit, QVBoxLayout, QWidget)

from .core import MODELS, STEMS, app_data, check_ffmpeg, export_stems, validate_audio, waveform_peaks


class Waveform(QWidget):
    seek = Signal(float)

    def __init__(self):
        super().__init__()
        self.peaks = np.zeros(800, dtype=np.float32)
        self.fraction = 0.0
        self.setMinimumHeight(165)

    def set_peaks(self, peaks):
        self.peaks = peaks
        self.fraction = 0.0
        self.update()

    def set_fraction(self, fraction):
        self.fraction = max(0.0, min(1.0, fraction))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.seek.emit(max(0.0, min(1.0, event.position().x() / max(1, self.width()))))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151925"))
        w, h = self.width(), self.height()
        painter.setPen(QPen(QColor("#242c3d"), 1))
        painter.drawLine(0, h // 2, w, h // 2)
        if len(self.peaks):
            painter.setPen(QPen(QColor("#8d79e7"), 1))
            for x in range(w):
                p = float(self.peaks[min(len(self.peaks) - 1, x * len(self.peaks) // max(1, w))])
                half = max(1, int(p * (h - 18) / 2))
                painter.drawLine(x, h // 2 - half, x, h // 2 + half)
        painter.setPen(QPen(QColor("#f4cc79"), 2))
        x = int(w * self.fraction)
        painter.drawLine(x, 0, x, h)
        painter.end()


class WaveformThread(QThread):
    ready = Signal(str, object)
    failed = Signal(str)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.ready.emit(str(self.path), waveform_peaks(self.path))
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportThread(QThread):
    progress = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, stems, source, model, destination, rate, depth):
        super().__init__()
        self.stems, self.source, self.model = stems, source, model
        self.destination, self.rate, self.depth = destination, rate, depth
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            result = export_stems(self.stems, self.source, self.model, self.destination,
                                  self.rate, self.depth, self.cancel_event,
                                  lambda i, n: self.progress.emit(i, n))
            self.succeeded.emit(str(result))
        except Exception as exc:
            self.failed.emit(str(exc))


def clock(ms):
    seconds = max(0, ms // 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SampleForge AI  |  MVP 0.1")
        self.resize(1100, 730)
        self.setAcceptDrops(True)
        self.source: Path | None = None
        self.current: Path | None = None
        self.stems: dict[str, Path] = {}
        self.model_used = MODELS[0]
        self.proc: QProcess | None = None
        self.proc_buffer = ""
        self.separation_error = ""
        self.lib_proc: QProcess | None = None
        self.lib_buffer = ""
        self.lib_error = ""
        self.lib_result: list[dict] = []
        self.lib_folder = ""
        self.export_thread: ExportThread | None = None
        self.wave_threads: list[WaveformThread] = []

        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.8)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio)
        self.player.positionChanged.connect(self.on_position)
        self.player.durationChanged.connect(self.on_duration)
        self.player.playbackStateChanged.connect(self.on_playback_state)
        self.player.errorOccurred.connect(lambda _error, message: self.log("Player: " + message))

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 22, 24, 22)
        outer.setSpacing(15)

        header = QHBoxLayout()
        brand = QLabel("SAMPLEFORGE  <span style='color:#a58bff'>AI</span>")
        brand.setStyleSheet("font-size: 26px; font-weight: 800; letter-spacing: 1px;")
        header.addWidget(brand)
        header.addStretch()
        self.advanced_toggle = QCheckBox("Advanced")
        self.advanced_toggle.toggled.connect(self.toggle_advanced)
        header.addWidget(self.advanced_toggle)
        outer.addLayout(header)

        self.drop = QPushButton("＋  Arraste um WAV, MP3 ou FLAC aqui  •  ou clique para escolher")
        self.drop.setObjectName("drop")
        self.drop.setMinimumHeight(84)
        self.drop.clicked.connect(self.open_audio)
        outer.addWidget(self.drop)

        controls = QHBoxLayout()
        self.generate = QPushButton("Separate 4 Stems")
        self.generate.setObjectName("primary")
        self.generate.clicked.connect(self.start_separation)
        self.generate.setEnabled(False)
        controls.addWidget(self.generate)
        self.extract_btn = QPushButton("Generate Sample Library")
        self.extract_btn.clicked.connect(self.start_library)
        self.extract_btn.setEnabled(False)
        controls.addWidget(self.extract_btn)
        self.export_btn = QPushButton("Export WAV Stems")
        self.export_btn.clicked.connect(self.start_export)
        self.export_btn.setEnabled(False)
        controls.addWidget(self.export_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_task)
        self.cancel_btn.setEnabled(False)
        controls.addWidget(self.cancel_btn)
        controls.addStretch()
        outer.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        outer.addWidget(self.progress)
        self.status = QLabel("Importe uma faixa para começar. Nenhum áudio é enviado à nuvem.")
        self.status.setObjectName("muted")
        outer.addWidget(self.status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame()
        left.setObjectName("panel")
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("SOURCE & STEMS"))
        self.category_filter = QComboBox()
        self.category_filter.addItems(["All", "kick", "snare", "hihat", "cymbals", "toms",
                                       "bass", "vocal", "drums loop", "bass loop",
                                       "vocals loop", "other loop"])
        self.category_filter.currentTextChanged.connect(self.filter_files)
        left_layout.addWidget(self.category_filter)
        self.files = QListWidget()
        self.files.currentItemChanged.connect(self.select_item)
        left_layout.addWidget(self.files)
        left_layout.addWidget(QLabel("Separação pode conter vazamento e artefatos."))
        splitter.addWidget(left)

        right = QFrame()
        right.setObjectName("panel")
        right_layout = QVBoxLayout(right)
        self.now = QLabel("Nenhum arquivo selecionado")
        self.now.setStyleSheet("font-size: 17px; font-weight: 650;")
        right_layout.addWidget(self.now)
        self.wave = Waveform()
        self.wave.seek.connect(lambda f: self.player.setPosition(int(self.player.duration() * f)))
        right_layout.addWidget(self.wave)
        transport = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        transport.addWidget(self.play_btn)
        stop = QPushButton("Stop")
        stop.clicked.connect(self.player.stop)
        transport.addWidget(stop)
        self.time = QLabel("00:00 / 00:00")
        transport.addWidget(self.time)
        transport.addStretch()
        right_layout.addLayout(transport)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.setRange(0, 0)
        self.position.sliderMoved.connect(self.player.setPosition)
        right_layout.addWidget(self.position)
        self.music_info = QLabel("BPM: —   •   Tonalidade: —   (análise musical na Etapa 2)")
        self.music_info.setObjectName("muted")
        right_layout.addWidget(self.music_info)
        right_layout.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([300, 700])
        outer.addWidget(splitter, 1)

        self.advanced_panel = QFrame()
        self.advanced_panel.setObjectName("panel")
        advanced = QVBoxLayout(self.advanced_panel)
        settings = QHBoxLayout()
        settings.addWidget(QLabel("4-stem model"))
        self.model = QComboBox()
        self.model.addItems(MODELS)
        settings.addWidget(self.model)
        settings.addWidget(QLabel("Device"))
        self.device = QComboBox()
        self.device.addItem("Auto (GPU / CPU)", "auto")
        self.device.addItem("Force CPU (low VRAM)", "cpu")
        settings.addWidget(self.device)
        settings.addWidget(QLabel("WAV rate"))
        self.rate = QComboBox()
        self.rate.addItems(["44100", "48000"])
        settings.addWidget(self.rate)
        settings.addWidget(QLabel("Bits"))
        self.depth = QComboBox()
        self.depth.addItems(["16", "24"])
        self.depth.setCurrentText("24")
        settings.addWidget(self.depth)
        advanced.addLayout(settings)
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumHeight(105)
        advanced.addWidget(self.logs)
        self.advanced_panel.hide()
        outer.addWidget(self.advanced_panel)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0c101a; color: #e9e9f2; font-size: 13px; }
            QPushButton { background: #252b3b; border: 1px solid #363d53; padding: 11px 16px;
                          border-radius: 9px; font-weight: 600; }
            QPushButton:hover { background: #363f57; }
            QPushButton:disabled { color: #74798a; background: #1b202d; }
            QPushButton#primary { background: #7255db; border-color: #8e74ee; }
            QPushButton#primary:hover { background: #876de6; }
            QPushButton#drop { background: #161b29; border: 2px dashed #56547e; color: #c5b8ff; }
            QFrame#panel { background: #151a27; border: 1px solid #2a3143; border-radius: 12px; }
            QListWidget, QTextEdit { background: #111622; border: 1px solid #31394d; border-radius: 7px; }
            QListWidget::item { padding: 10px; }
            QListWidget::item:selected { background: #51448b; }
            QComboBox { background: #22283a; padding: 6px; border-radius: 6px; }
            QLabel#muted { color: #a4aabc; }
            QProgressBar { background: #22283a; border: none; border-radius: 4px; height: 7px; }
            QProgressBar::chunk { background: #a58bff; border-radius: 4px; }
        """)

    def toggle_advanced(self, enabled):
        self.advanced_panel.setVisible(enabled)

    def log(self, message):
        self.logs.append(message)
        self.status.setText(message[:250])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(u.isLocalFile() and Path(u.toLocalFile()).suffix.lower()
                                                  in (".wav", ".mp3", ".flac") for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in (".wav", ".mp3", ".flac"):
                self.import_audio(Path(url.toLocalFile()))
                event.acceptProposedAction()
                break

    def open_audio(self):
        name, _ = QFileDialog.getOpenFileName(self, "Abrir música", "", "Audio (*.wav *.mp3 *.flac)")
        if name:
            self.import_audio(Path(name))

    def import_audio(self, path):
        if self.is_busy():
            QMessageBox.warning(self, "Processando", "Cancele a tarefa antes de importar outra faixa.")
            return
        try:
            check_ffmpeg()
            self.source = validate_audio(path)
        except Exception as exc:
            QMessageBox.critical(self, "Erro ao importar", str(exc))
            return
        self.stems = {}
        self.lib_result = []
        self.lib_folder = ""
        self.extract_btn.setEnabled(False)
        self.category_filter.setCurrentText("All")
        self.files.clear()
        self.add_item("Original  •  " + self.source.name, self.source)
        self.files.setCurrentRow(0)
        self.generate.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log(f"Importado: {self.source.name}")

    def add_item(self, name, path, category="source"):
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setData(Qt.ItemDataRole.UserRole + 1, category)
        self.files.addItem(item)
        self.filter_files(self.category_filter.currentText())

    def filter_files(self, category):
        for i in range(self.files.count()):
            item = self.files.item(i)
            item.setHidden(category != "All" and item.data(Qt.ItemDataRole.UserRole + 1) != category)

    def select_item(self, current, _previous):
        if not current:
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        self.current = path
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.now.setText(current.text())
        self.wave.set_peaks(np.zeros(800, dtype=np.float32))
        thread = WaveformThread(path)
        self.wave_threads.append(thread)
        thread.ready.connect(self.on_waveform)
        thread.failed.connect(lambda error: self.log("Waveform: " + error))
        thread.finished.connect(lambda t=thread: self.wave_threads.remove(t) if t in self.wave_threads else None)
        thread.start()

    def on_waveform(self, path, peaks):
        if self.current and str(self.current) == path:
            self.wave.set_peaks(peaks)

    def on_position(self, pos):
        if not self.position.isSliderDown():
            self.position.setValue(pos)
        duration = self.player.duration()
        self.time.setText(f"{clock(pos)} / {clock(duration)}")
        self.wave.set_fraction(pos / duration if duration else 0)

    def on_duration(self, duration):
        self.position.setRange(0, duration)
        self.on_position(self.player.position())

    def on_playback_state(self, state):
        self.play_btn.setText("Pause" if state == QMediaPlayer.PlaybackState.PlayingState else "Play")

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def is_busy(self):
        return (self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning) or (
            self.lib_proc is not None and self.lib_proc.state() != QProcess.ProcessState.NotRunning) or (
            self.export_thread is not None and self.export_thread.isRunning())

    def start_separation(self):
        if not self.source or self.is_busy():
            return
        self.stems = {}
        self.extract_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.category_filter.setCurrentText("All")
        self.files.clear()
        self.add_item("Original  •  " + self.source.name, self.source)
        self.files.setCurrentRow(0)
        self.model_used = self.model.currentText()
        job = app_data() / "jobs" / uuid.uuid4().hex
        job.mkdir(parents=True)
        self.separation_error = ""
        self.proc_buffer = ""
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.read_process)
        self.proc.finished.connect(self.separation_finished)
        self.proc.errorOccurred.connect(lambda error: self.log(f"Processo: {error}"))
        self.proc.start(sys.executable, ["-u", "-m", "sampleforge.worker", "--input",
                                         str(self.source), "--job", str(job), "--model", self.model_used,
                                         "--device", self.device.currentData()])
        self.progress.setRange(0, 0)
        self.generate.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log("Iniciando separação local...")

    def read_process(self):
        if not self.proc:
            return
        self.proc_buffer += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        while "\n" in self.proc_buffer:
            line, self.proc_buffer = self.proc_buffer.split("\n", 1)
            if line.startswith("SF_EVENT:"):
                try:
                    event = json.loads(line[len("SF_EVENT:"):])
                except json.JSONDecodeError:
                    self.log("Evento inválido do worker")
                    continue
                if event.get("type") == "status":
                    self.log(event.get("message", "Processando..."))
                elif event.get("type") == "error":
                    self.separation_error = event.get("message", "Falha desconhecida")
                    self.log(self.separation_error)
                elif event.get("type") == "result":
                    self.stems = {k: Path(v) for k, v in event.get("stems", {}).items()}
            elif line.strip():
                self.logs.append(line[:600])

    def separation_finished(self, exit_code, _status):
        self.read_process()
        self.progress.setRange(0, 100)
        self.cancel_btn.setEnabled(False)
        self.generate.setEnabled(self.source is not None)
        if exit_code == 0 and set(self.stems) == set(STEMS) and all(p.is_file() for p in self.stems.values()):
            for stem in STEMS:
                self.add_item(stem.upper() + "  •  WAV", self.stems[stem])
            self.export_btn.setEnabled(True)
            self.extract_btn.setEnabled(True)
            self.progress.setValue(100)
            self.log("Quatro stems gerados. Selecione cada stem para ouvir.")
        else:
            self.stems = {}
            self.progress.setValue(0)
            message = self.separation_error or f"Processamento interrompido (código {exit_code}). Consulte Advanced."
            self.log(message)
            if not self.separation_error and exit_code != 0:
                self.logs.append("Se houver erro de CUDA/memória, tente a instalação CPU.")
        self.proc.deleteLater()
        self.proc = None

    def start_library(self):
        if not self.stems or not self.source or self.is_busy():
            return
        folder = QFileDialog.getExistingDirectory(self, "Pasta para gerar a biblioteca de samples")
        if not folder:
            return
        self.lib_buffer, self.lib_error, self.lib_result, self.lib_folder = "", "", [], ""
        self.lib_proc = QProcess(self)
        self.lib_proc.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self.lib_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.lib_proc.readyReadStandardOutput.connect(self.read_library)
        self.lib_proc.finished.connect(self.library_finished)
        self.lib_proc.start(sys.executable, ["-u", "-m", "sampleforge.extract_worker",
                                          "--source", str(self.source),
                                          "--job", str(self.stems["drums"].parent),
                                          "--destination", folder])
        self.progress.setRange(0, 0)
        self.cancel_btn.setEnabled(True)
        self.generate.setEnabled(False)
        self.extract_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.log("Separando elementos da bateria e criando WAVs individuais...")

    def read_library(self):
        if not self.lib_proc:
            return
        self.lib_buffer += bytes(self.lib_proc.readAllStandardOutput()).decode("utf-8", "replace")
        while "\n" in self.lib_buffer:
            line, self.lib_buffer = self.lib_buffer.split("\n", 1)
            if line.startswith("SF_EVENT:"):
                try:
                    event = json.loads(line[len("SF_EVENT:"):])
                except json.JSONDecodeError:
                    self.log("Evento inválido do gerador de samples.")
                    continue
                if event.get("type") == "status":
                    self.log(event.get("message", "Processando samples..."))
                elif event.get("type") == "error":
                    self.lib_error = event.get("message", "Falha na extração")
                    self.log(self.lib_error)
                elif event.get("type") == "result":
                    self.lib_folder = event.get("folder", "")
                    self.lib_result = event.get("samples", [])
            elif line.strip():
                self.logs.append(line[:600])

    def library_finished(self, code, _status):
        self.read_library()
        self.progress.setRange(0, 100)
        self.cancel_btn.setEnabled(False)
        self.generate.setEnabled(self.source is not None)
        self.extract_btn.setEnabled(bool(self.stems))
        self.export_btn.setEnabled(bool(self.stems))
        if code == 0 and self.lib_result and all(Path(rec["path"]).is_file() for rec in self.lib_result):
            self.category_filter.setCurrentText("All")
            for rec in self.lib_result:
                self.add_item(str(rec["category"]).upper() + " • " + str(rec["name"]),
                              Path(rec["path"]), str(rec["category"]))
            self.progress.setValue(100)
            self.log(f"Biblioteca gerada: {len(self.lib_result)} WAVs em {self.lib_folder}")
            QMessageBox.information(self, "Sample Library",
                f"{len(self.lib_result)} arquivos WAV gerados!\n{self.lib_folder}\n\n"
                "A separação e o BPM são estimados. Ouça e revise os recortes.")
        else:
            self.progress.setValue(0)
            self.log(self.lib_error or f"Extração interrompida (código {code}). Veja Advanced.")
        self.lib_proc.deleteLater()
        self.lib_proc = None

    def start_export(self):
        if not self.stems or not self.source or self.is_busy():
            return
        folder = QFileDialog.getExistingDirectory(self, "Escolha a pasta para exportar")
        if not folder:
            return
        self.export_thread = ExportThread(dict(self.stems), self.source, self.model_used,
                                          Path(folder), int(self.rate.currentText()), int(self.depth.currentText()))
        self.export_thread.progress.connect(lambda i, n: self.progress.setValue(int(100 * i / n)))
        self.export_thread.succeeded.connect(self.export_success)
        self.export_thread.failed.connect(self.export_failure)
        self.export_thread.finished.connect(self.export_finished)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.cancel_btn.setEnabled(True)
        self.generate.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.log("Exportando WAVs...")
        self.export_thread.start()

    def export_success(self, path):
        self.log("Exportado para: " + path)
        QMessageBox.information(self, "Exportação concluída", path)

    def export_failure(self, error):
        self.log("Exportação: " + error)
        QMessageBox.warning(self, "Exportação interrompida", error)

    def export_finished(self):
        self.cancel_btn.setEnabled(False)
        self.generate.setEnabled(self.source is not None)
        self.export_btn.setEnabled(bool(self.stems))
        if self.export_thread:
            self.export_thread.deleteLater()
            self.export_thread = None

    def cancel_task(self):
        if self.lib_proc and self.lib_proc.state() != QProcess.ProcessState.NotRunning:
            self.log("Cancelando geração da biblioteca...")
            proc = self.lib_proc
            proc.terminate()
            QTimer.singleShot(3500, lambda: proc.kill() if proc.state() != QProcess.ProcessState.NotRunning else None)
        if self.proc and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.log("Cancelando separação...")
            proc = self.proc
            proc.terminate()
            QTimer.singleShot(3500, lambda: proc.kill() if proc.state() != QProcess.ProcessState.NotRunning else None)
        if self.export_thread and self.export_thread.isRunning():
            self.log("Cancelando exportação...")
            self.export_thread.cancel()

    def closeEvent(self, event):
        if self.is_busy():
            QMessageBox.warning(self, "Tarefa ativa", "Cancele o processamento e aguarde antes de sair.")
            event.ignore()
            return
        if any(t.isRunning() for t in self.wave_threads):
            QMessageBox.information(self, "Aguarde", "Aguarde a leitura da waveform antes de fechar.")
            event.ignore()
            return
        self.player.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
