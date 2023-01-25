/* DECLARACION DE CONSTANTES Y VARIABLES GLOBALES */
int dato; // Byte de información recibido por el puerto serie

// Decharación de pines a utilizar
#define PIN_BUZZER  13  // Definir el pin de salida del BUZZER (zumbador)
#define PIN_UP      9   // Definir el pin del vibrador que indica hacia arriba
#define PIN_DOWN    10  // Definir el pin del vibrador que indica hacia abajo
#define PIN_LEFT    11  // Definir el pin del vibrador que indica hacia la izquierda
#define PIN_RIGHT   12  // Definir el pin del vibrador que indica hacia la derecha

// Decharación frecuencia y duración de la nota musical a utilizar mediante vibraciones y el BUZZER
#define NOTE_C3     130 // Definir la frecuencia de la nota C3 (130 Hz)
#define DURATION    75  // Definir la duración de las notas musicales en milisegundos

void setup() {
  Serial.begin(9600, SERIAL_8N1); // opens serial port, sets data rate to 9600 bps
  pinMode(PIN_BUZZER, OUTPUT);    // Definir el pin del BUZZER como salida
  pinMode(PIN_UP, OUTPUT);        // Definir el pin del vibrador que indica hacia arriba como salida
  pinMode(PIN_DOWN, OUTPUT);      // Definir el pin del vibrador que indica hacia abajo como salida
  pinMode(PIN_LEFT, OUTPUT);      // Definir el pin del vibrador que indica hacia la izquierda como salida
  pinMode(PIN_RIGHT, OUTPUT);     // Definir el pin del vibrador que indica hacia la derecha como salida
}

void loop() {
    // Comunicación serial
    if (Serial.available() > 0) { // Si hay datos en el puerto serie
        dato = Serial.read();   // Lee el dato enviado por el puerto serie usando código ASCII
        // Encendido del BUZZER
        if (dato == 48){ // Si el dato enviado es 0 (48 codificado en ASCII) 
            tone(PIN_BUZZER, NOTE_C3, DURATION);
        }
        // Comandos del encendido del la interfáz háptica {100, 108, 114, 117} => {down, left, right, up}
        if (dato == 100){ // Si el dato resivido es d (codificado en ASCII)
            tone(PIN_DOWN, NOTE_C3, DURATION);
        }
        if (dato == 108){ // Si el dato resivido es l (codificado en ASCII)
            tone(PIN_LEFT, NOTE_C3, DURATION);
        }
        if (dato == 114){ // Si el dato resivido es r (codificado en ASCII)
            tone(PIN_RIGHT, NOTE_C3, DURATION);
        }
        if (dato == 117){ // Si el dato resivido es u (codificado en ASCII)
            tone(PIN_UP, NOTE_C3, DURATION);
        }
    }
}