from ultralytics import YOLO
import cv2
from brain import generateSpeak
import pyttsx3
from loguru import logger
from datetime import datetime, timedelta
from threading import Thread
from queue import Queue

def check_expire_time(obj, chave, min_interval_seconds=10):
    if chave in obj:
        time_since_last = datetime.now() - obj[chave]
        if time_since_last < timedelta(seconds=min_interval_seconds):
            return False
    return True

class SpeechEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.speech_queue = Queue()
        self.is_speaking = False
        
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'pt' in voice.languages:
                self.engine.setProperty('voice', voice.id)
                break
        
        self.speech_thread = Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
    
    def _speech_worker(self):
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_speaking = False
    
    def say(self, text):
        self.speech_queue.put(text)
    
    def stop(self):
        self.speech_queue.put(None)
        self.speech_thread.join()

def initialize():
    logger = logger.bind(name="ultralytics")
    logger.disable("ultralytics")
    logger.level("ERROR", no=40)

    # Dicionário para armazenar o último momento em que cada objeto foi detectado em cada posição
    last_detections = {}

    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)
    speech_engine = SpeechEngine()

    CONFIDENCE_THRESHOLD = 0.7

    _, frame = cap.read()
    HEIGHT, WIDTH = frame.shape[:2]
    CENTER_X = WIDTH // 2
    CENTER_Y = HEIGHT // 2

    print(f"Tamanho da tela: {WIDTH}x{HEIGHT}")
    print(f"Centro: ({CENTER_X}, {CENTER_Y})")

    def get_direction(x, y):
        directions = []
        
        margin_x = WIDTH // 3
        margin_y = HEIGHT // 3
        
        if x < margin_x:
            directions.append("esquerda")
        elif x > (WIDTH - margin_x):
            directions.append("direita")
        else:
            directions.append("centro")
            
        if y < margin_y:
            directions.append("cima")
        elif y > (HEIGHT - margin_y):
            directions.append("baixo")
        else:
            directions.append("frente")
            
        return directions

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = frame.copy()
            
            # Desenhar linhas de referência
            cv2.line(annotated_frame, (WIDTH//3, 0), (WIDTH//3, HEIGHT), (100, 100, 100), 1)
            cv2.line(annotated_frame, (2*WIDTH//3, 0), (2*WIDTH//3, HEIGHT), (100, 100, 100), 1)
            cv2.line(annotated_frame, (0, HEIGHT//3), (WIDTH, HEIGHT//3), (100, 100, 100), 1)
            cv2.line(annotated_frame, (0, 2*HEIGHT//3), (WIDTH, 2*HEIGHT//3), (100, 100, 100), 1)
            
            phrases = []

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    
                    if conf > CONFIDENCE_THRESHOLD:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        obj_center_x = (x1 + x2) // 2
                        obj_center_y = (y1 + y2) // 2
                        
                        direction = get_direction(obj_center_x, obj_center_y)
                        
                        # Criar uma chave única para cada objeto em cada posição
                        detection_key = f"{class_name}_{direction[0]}_{direction[1]}".lower()

                        # Verificar se deve anunciar novamente
                        if not speech_engine.is_speaking and check_expire_time(last_detections, detection_key, 10):
                            speak = generateSpeak(direction, class_name)
                            phrases.append(speak)
                            # Atualizar o timestamp da última detecção
                            last_detections[detection_key] = datetime.now()
                        
                        # Desenhar retângulo e informações
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(annotated_frame, (obj_center_x, obj_center_y), 4, (0, 0, 255), -1)
                        
                        # Adicionar texto com informações
                        info = f"{class_name}: {conf:.2f}"
                        direction_text = f"Direcao: {' '.join(direction)}"
                        
                        # Se estiver em cooldown, adicionar essa informação
                        if detection_key in last_detections:
                            time_remaining = 10 - (datetime.now() - last_detections[detection_key]).seconds
                            if time_remaining > 0:
                                direction_text += f" (cooldown: {time_remaining}s)"
                        
                        cv2.putText(annotated_frame, info, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(annotated_frame, direction_text, (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if phrases:
                speech_engine.say(', '.join(phrases))

            cv2.imshow('Webcam com YOLO', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        speech_engine.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    initialize()