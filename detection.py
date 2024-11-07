from ultralytics import YOLO
import cv2
from brain import generateSpeak
import pyttsx3
import asyncio
import logging
from datetime import datetime, timedelta

def check_expire_time(obj, chave):
    print(obj)
    if chave in obj:
        # Calcula a diferença de tempo
        time = datetime.now() - obj[chave]
        print(time, timedelta(seconds=3))
        # Verifica se o tempo é maior que 3 segundos
        if time > timedelta(seconds=3):
            return False
    return True

def initialize():
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    # Local onde ficará armazenado temporariamente as coisas.
    last_speak = {}

    # Inicializar o modelo YOLO
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(0)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'pt' in voice.languages: 
            engine.setProperty('voice', voice.id)
            break

    CONFIDENCE_THRESHOLD = 0.7

    # Pegar o primeiro frame para ter as dimensões
    _, frame = cap.read()
    HEIGHT, WIDTH = frame.shape[:2]
    CENTER_X = WIDTH // 2
    CENTER_Y = HEIGHT // 2

    print(f"Tamanho da tela: {WIDTH}x{HEIGHT}")
    print(f"Centro: ({CENTER_X}, {CENTER_Y})")

    def get_direction(x, y):
        """Determina a direção do objeto em relação ao centro"""
        directions = []
        
        # Divide a tela em 3 partes horizontais e verticais
        margin_x = WIDTH // 3
        margin_y = HEIGHT // 3
        
        # Direção horizontal
        if x < margin_x:
            directions.append("esquerda")
        elif x > (WIDTH - margin_x):
            directions.append("direita")
        else:
            directions.append("centro")
            
        # Direção vertical
        if y < margin_y:
            directions.append("cima")
        elif y > (HEIGHT - margin_y):
            directions.append("baixo")
        else:
            directions.append("frente")
            
        return directions

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
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                
                if conf > CONFIDENCE_THRESHOLD:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calcular o centro do objeto
                    obj_center_x = (x1 + x2) // 2
                    obj_center_y = (y1 + y2) // 2
                    
                    # Pegar a direção
                    direction = get_direction(obj_center_x, obj_center_y)

                    key = f"{class_name}_{direction[0]}_{direction[1]}".lower()

                    speaking = False;

                    if (speaking == False and check_expire_time(last_speak, key)):
                        def on_end():
                            nonlocal speaking
                            speaking = True;


                        async def speakFunc():
                            nonlocal speak, engine, on_end
                            engine.say(speak)
                            engine.runAndWait()
                            engine.connect('finished-utterance', on_end)
                        
                        speak = generateSpeak(direction, class_name)
                        asyncio.run(speakFunc())
                        speaking = True
                        last_speak[key] = datetime.now()
                        
                        
                    
                    # Desenhar
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated_frame, (obj_center_x, obj_center_y), 4, (0, 0, 255), -1)
                    
                    # Mostrar informações
                    info = f"{class_name}: {conf:.2f}"
                    direction_text = f"Direcao: {' '.join(direction)}"
                    cv2.putText(annotated_frame, info, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, direction_text, (x1, y1 - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Webcam com YOLO', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()