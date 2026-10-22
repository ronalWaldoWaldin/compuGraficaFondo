import cv2
import numpy as np

# ----------------------------
# Cargar fondo de reemplazo
# ----------------------------
bg = cv2.imread("fondo.jpg")

if bg is None:
    print("No se encontró fondo.jpg")
    exit()

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

fondo_referencia = None

print("Presiona ESPACIO para capturar el fondo de referencia")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ---------------------------------
    # Captura inicial del fondo
    # ---------------------------------
    if fondo_referencia is None:

        cv2.putText(
            frame,
            "ESPACIO = Capturar fondo",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Resultado", frame)

        key = cv2.waitKey(1)

        if key == 32:  # espacio
            fondo_referencia = frame.copy()
            print("Fondo capturado")

        elif key == 27:
            break

        continue

    # ---------------------------------
    # Redimensionar fondo de reemplazo
    # ---------------------------------
    fondo_nuevo = cv2.resize(
        bg,
        (frame.shape[1], frame.shape[0])
    )

    # ---------------------------------
    # Suavizado para reducir ruido
    # ---------------------------------
    frame_blur = cv2.GaussianBlur(
        frame,
        (9, 9),
        0
    )

    fondo_blur = cv2.GaussianBlur(
        fondo_referencia,
        (9, 9),
        0
    )

    # ---------------------------------
    # Diferencia absoluta
    # ---------------------------------
    diff = cv2.absdiff(
        frame_blur,
        fondo_blur
    )

    # ---------------------------------
    # Escala de grises
    # ---------------------------------
    gray = cv2.cvtColor(
        diff,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------
    # Threshold
    # ---------------------------------
    _, mask = cv2.threshold(
        gray,
        15,
        255,
        cv2.THRESH_BINARY
    )

    # ---------------------------------
    # Morfología
    # ---------------------------------
    kernel = np.ones((5, 5), np.uint8)

    # quitar ruido
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # rellenar huecos
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=3
    )

    # ---------------------------------
    # Mantener solo el mayor objeto
    # ---------------------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mask_limpia = np.zeros_like(mask)

    if contours:

        mayor = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(mayor)

        # evita objetos pequeños
        if area > 8000:

            cv2.drawContours(
                mask_limpia,
                [mayor],
                -1,
                255,
                thickness=cv2.FILLED
            )

    # ---------------------------------
    # Suavizar bordes
    # ---------------------------------
    mask_limpia = cv2.GaussianBlur(
        mask_limpia,
        (11, 11),
        0
    )

    # ---------------------------------
    # Máscara invertida
    # ---------------------------------
    mask_f = mask_limpia.astype(np.float32) / 255.0
    mask_f = cv2.merge([mask_f, mask_f, mask_f])

    # ---------------------------------
    # Composición final
    # ---------------------------------
    foreground = frame.astype(np.float32) * mask_f

    background = fondo_nuevo.astype(np.float32) * (1 - mask_f)

    resultado = cv2.add(
        foreground,
        background
    )

    resultado = resultado.astype(np.uint8)

    # ---------------------------------
    # Contorno verde para depuración
    # ---------------------------------
    if contours:
        cv2.drawContours(
            resultado,
            [mayor],
            -1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Resultado", resultado)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()