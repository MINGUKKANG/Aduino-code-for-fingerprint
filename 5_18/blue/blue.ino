#include "SoftwareSerial.h"
SoftwareSerial BTSerial(9,8);
char val;

void setup() {
  Serial.begin(9600);
  BTSerial.begin(9600);
}

void loop() {
  if (BTSerial.available()){
    val = BTSerial.read();
    Serial.write(val);
  }
  if (Serial.available()){
    BTSerial.write(Serial.read());
  }
}
