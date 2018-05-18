/* 
  FPS_Enroll.ino - Library example for controlling the GT-511C3 Finger Print Scanner (FPS)
  Created by Josh Hawley, July 23rd 2013
  Licensed for non-commercial use, must include this license message
  basically, Feel free to hack away at it, but just give me credit for my work =)
  TLDR; Wil Wheaton's Law

  This sketch will attempt to identify a previously enrolled fingerprint.
*/
#include "FPS_GT511C3.h"
#include "SoftwareSerial.h"
// Hardware setup - FPS connected to:
// digital pin 2(arduino rx, fps tx)
// digital pin 3(arduino tx - 560ohm resistor fps tx - 1000ohm resistor - ground)
// this brings the 5v tx line down to about 3.2v so we dont fry our fps
byte val;


FPS_GT511C3 fps(2, 3);
SoftwareSerial BTSerial(9,8);

void setup() {
  Serial.begin(9600);
  BTSerial.begin(9600);
  delay(100);
  pinMode(4, OUTPUT);
  pinMode(6, OUTPUT);
  fps.Open();
  fps.SetLED(true);
}

void loop() {
  if (BTSerial.available()){
    byte val = BTSerial.read();
    Serial.write(val);}
    
  if ((val=='O')||(val=='o')){
    tone(6,261);
    delay(500);
    tone(6,294);
    delay(500);
    tone(6,330);
    delay(500);
    noTone(6);
    digitalWrite(4,HIGH);
    delay(1500);
    digitalWrite(4,LOW);
    delay(1500);
  }

  // Identify fingerprint test
  if (fps.IsPressFinger()) {
    fps.CaptureFinger(false);
    int id = fps.Identify1_N();
    Serial.println(id);
    if (id <200) {
      Serial.print("Verified ID:");
      Serial.println(id);
      
      tone(6,261);
      delay(500);
      tone(6,294);
      delay(500);
      tone(6,330);
      delay(500);
      noTone(6);
      
      digitalWrite(4,HIGH);
      delay(1500);
      digitalWrite(4,LOW);
      delay(1500);
      }else {
      tone(6,349);
      delay(500);
      tone(6,349);
      delay(500);
      noTone(6);}
  }
}
