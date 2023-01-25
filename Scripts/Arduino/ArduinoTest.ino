int dato; 

// Decharación de pines a utilizar
#define PIN_BUZZER  11  // Definir el pin de salida del BUZZER
#define PIN_UP      2   // Definir el pin de salida del BUZZER
#define PIN_DOWN    3   // Definir el pin de salida del BUZZER
#define PIN_LEFT    4   // Definir el pin de salida del BUZZER
#define PIN_RIGHT   5   // Definir el pin de salida del BUZZER

// Decharación frecuencias y duración de notas musicales a utilizar mediante el BUZZER
#define NOTE_C3     130 // Definir la frecuencia de la nota C3 (130 Hz)
#define NOTE_D3     147 // Definir la frecuencia de la nota D3 (147 Hz)
#define NOTE_E3     165 // Definir la frecuencia de la nota E3 (165 Hz)
#define NOTE_F3     175 // Definir la frecuencia de la nota F3 (175 Hz)
#define NOTE_G3     196 // Definir la frecuencia de la nota G3 (196 Hz)
#define NOTE_A3     220 // Definir la frecuencia de la nota A3 (220 Hz)
#define NOTE_B3     247 // Definir la frecuencia de la nota B3 (247 Hz)
#define NOTE_C4     262 // Definir la frecuencia de la nota C4 (262 Hz)
#define NOTE_D4     294 // Definir la frecuencia de la nota D4 (294 Hz)
#define NOTE_E4     330 // Definir la frecuencia de la nota E4 (330 Hz)
#define NOTE_F4     349 // Definir la frecuencia de la nota F4 (349 Hz)
#define NOTE_G4     392 // Definir la frecuencia de la nota G4 (392 Hz)
#define NOTE_A4     440 // Definir la frecuencia de la nota A4 (440 Hz)
#define NOTE_B4     494 // Definir la frecuencia de la nota B4 (494 Hz)
#define DURATION    100  // Definir la duración de las notas musicales en milisegundos


void setup() {
  Serial.begin(9600, SERIAL_8N1); // opens serial port, sets data rate to 9600 bps
  pinMode(PIN_BUZZER, OUTPUT);
  
  // Melodía de inicio
  // tone(PIN_BUZZER, 880, 150); delay(50); tone(PIN_BUZZER, 440, 150);
  digitalWrite(PIN_BUZZER, LOW);
}

void loop() {
    // Comunicación serial
    if (Serial.available() > 0) { // Si hay datos en el puerto serie
        
        dato = Serial.read();   // Lee el dato enviado por el puerto serie usando código ASCII
        Serial.write(dato);     // Envia el dato por el puerto serie codificado en ASCII

        // Comandos de control del la interfáz háptica {100, 108, 114, 117} => {down, left, right, up}
        if (dato == 100){ // Si el dato resivido es d (codificado en ASCII)
            tone(PIN_DOWN, NOTE_C3, DURATION);
        }
        if (dato == 108){ // Si el dato resivido es l (codificado en ASCII)
            tone(PIN_LEFT, NOTE_E3, DURATION);
        }
        if (dato == 114){ // Si el dato resivido es r (codificado en ASCII)
            tone(PIN_RIGHT, NOTE_G3, DURATION);
        }
        if (dato == 117){ // Si el dato resivido es u (codificado en ASCII)
            tone(PIN_UP, NOTE_B3, DURATION);
        }

        // Comandos de control del la interfáz sonora
        if (dato == 48){ // Si el dato enviado es 0 (48 codificado en ASCII) 
            tone(PIN_BUZZER, NOTE_C3, DURATION);
        }
        if (dato == 49){ // Si el dato enviado es 1 (49 codificado en ASCII) 
            tone(PIN_BUZZER, NOTE_D3, DURATION);
        }
        if (dato == 50){ // Si el dato enviado es 2 (50 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_E3, DURATION);
        }
        if (dato == 51){ // Si el dato enviado es 3 (51 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_F3, DURATION);
        }
        if (dato == 52){ // Si el dato enviado es 4 (52 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_G3, DURATION);
        }
        if (dato == 53){ // Si el dato enviado es 5 (53 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_A3, DURATION);
        }
        if (dato == 54){ // Si el dato enviado es 6 (54 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_B3, DURATION);
        }
        if (dato == 55){ // Si el dato enviado es 7 (55 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_C4, DURATION);
        }
        if (dato == 56){ // Si el dato enviado es 8 (56 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_D4, DURATION);
        }
        if (dato == 57){ // Si el dato enviado es 9 (57 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_E4, DURATION);
        }
        if (dato == 65){ // Si el dato enviado es A (65 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_F4, DURATION);
        }
        if (dato == 66){ // Si el dato enviado es B (66 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_G4, DURATION);
        }
        if (dato == 67){ // Si el dato enviado es C (67 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_A4, DURATION);
        }
        if (dato == 68){ // Si el dato enviado es D (68 codificado en ASCII)
            tone(PIN_BUZZER, NOTE_B4, DURATION);
        }
    }
}