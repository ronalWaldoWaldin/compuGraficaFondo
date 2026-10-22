# SC3321 Vision Computacional : 584431 / 696641 : Xavier Maximiliano López Ibarra / Ronal Waldo Mora Aybar
"""
Proyecto Reemplazo de Fondo

Usage:
    main.py --bg=<imagen>

Opctions:
    --bg=<imagen>    Ruta de la imagen de fondo.
"""

from docopt import docopt
import cv2
import numpy as np


def main():

    args = docopt(__doc__)

    ruta_fondo = args["--bg"]

    fondo_nuevo = cv2.imread(ruta_fondo)

    if fondo_nuevo is None:
        print(f"No se pudo cargar: {ruta_fondo}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    fondo_referencia = None

    print("Presiona ESPACIO para capturar el fondo")
    print("Presiona ESC para salir")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ------------------------
        # Captura del fondo
        # ------------------------
        if fondo_referencia is None:

            vista = frame.copy()

            cv2.putText(
                vista,
                "ESPACIO = Capturar fondo",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Resultado", vista)

            key = cv2.waitKey(1)

            if key == 32:
                fondo_referencia = frame.copy()
                print("Fondo capturado")

            elif key == 27:
                break

            continue

        # ------------------------
        # Redimensionar fondo nuevo
        # ------------------------
        fondo_resize = cv2.resize(
            fondo_nuevo,
            (frame.shape[1], frame.shape[0])
        )

        # ------------------------
        # Suavizado
        # ------------------------
        frame_blur = cv2.GaussianBlur(
            frame,
            (11, 11),
            0
        )

        fondo_blur = cv2.GaussianBlur(
            fondo_referencia,
            (11, 11),
            0
        )

        # ------------------------
        # Diferencia
        # ------------------------
        diff = cv2.absdiff(
            frame_blur,
            fondo_blur
        )

        # ------------------------
        # Máximo entre canales RGB
        # ------------------------
        b, g, r = cv2.split(diff)

        gray = np.maximum(b, g)
        gray = np.maximum(gray, r)

        # ------------------------
        # Umbral
        # ------------------------
        _, mask = cv2.threshold(
            gray,
            30,
            255,
            cv2.THRESH_BINARY
        )

        # ------------------------
        # Morfología
        # ------------------------
        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=2
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        # ------------------------
        # Contornos
        # ------------------------
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        mask_limpia = np.zeros_like(mask)

        if len(contours) > 0:

            mayor = max(
                contours,
                key=cv2.contourArea
            )

            area = cv2.contourArea(mayor)

            if area > 5000:

                cv2.drawContours(
                    mask_limpia,
                    [mayor],
                    -1,
                    255,
                    cv2.FILLED
                )

        # ------------------------
        # Suavizar bordes
        # ------------------------
        mask_limpia = cv2.GaussianBlur(
            mask_limpia,
            (9, 9),
            0
        )

        # ------------------------
        # Máscara flotante
        # ------------------------
        mask_float = (
            mask_limpia.astype(np.float32) / 255.0
        )

        mask_float = cv2.merge([
            mask_float,
            mask_float,
            mask_float
        ])

        # ------------------------
        # Composición
        # ------------------------
        foreground = (
            frame.astype(np.float32)
            * mask_float
        )

        background = (
            fondo_resize.astype(np.float32)
            * (1.0 - mask_float)
        )

        resultado = cv2.add(
            foreground,
            background
        )

        resultado = resultado.astype(
            np.uint8
        )

        # ------------------------
        # Ventanas de depuración
        # ------------------------
        cv2.imshow(
            "Mascara",
            mask_limpia
        )

        cv2.imshow(
            "Resultado",
            resultado
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()