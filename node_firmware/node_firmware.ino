#include <WiFi.h>
#include <HTTPClient.h>
#include "DHT.h"

// ------------------------- EDIT THESE -------------------------
#define NODE_ID        "B"                       // "A", "B" or "C"
const char* WIFI_SSID  = "LAVZARO Network";       // 2.4 GHz only (ESP32 has no 5 GHz)
const char* WIFI_PASS  = "00000002";
const char* SERVER_URL = "http://10.81.218.109:5000/api/reading";   // laptop IP, port 5000
// --------------------------------------------------------------

#define MQ_PIN          34
#define DHT_PIN         4
#define DIVIDER_RATIO   0.6f     // 15k / (10k + 15k). Change if you use other resistors.
#define SEND_PERIOD_MS  300      // Ultra-responsive: sends 3-4 times per second (was 1000ms)
#define WARMUP_MS       60000UL  // MQ heater warm-up (use 5-10 min before real tests)

DHT dht(DHT_PIN, DHT11);
unsigned long lastSend = 0;
unsigned long lastDhtRead = 0;
float cachedTemp = 27.0;
float cachedHum  = 60.0;

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("Connecting to %s ", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) { delay(300); Serial.print("."); }
  if (WiFi.status() == WL_CONNECTED) Serial.printf("\nConnected, IP %s\n", WiFi.localIP().toString().c_str());
  else Serial.println("\nWi-Fi failed, will retry");
}

// Fast ADC read: 16 samples with microsecond delay (<1 ms total vs old 64 ms!)
float readMqMilliVolts() {
  uint32_t sum = 0;
  for (int i = 0; i < 16; i++) {
    sum += analogReadMilliVolts(MQ_PIN);
    delayMicroseconds(150);
  }
  return (sum / 16.0f) / DIVIDER_RATIO;
}

void setup() {
  Serial.begin(115200);
  analogSetPinAttenuation(MQ_PIN, ADC_11db);
  dht.begin();
  connectWiFi();
  Serial.printf("Node %s warming up MQ-135 ...\n", NODE_ID);
}

void loop() {
  connectWiFi();
  if (millis() - lastSend < SEND_PERIOD_MS) return;
  lastSend = millis();

  // Fast MQ-135 read (sub-millisecond)
  float mv = readMqMilliVolts();

  // Non-blocking DHT read every 3 seconds (DHT11 is slow ~250ms, so do NOT read every 300ms)
  if (millis() - lastDhtRead > 3000) {
    lastDhtRead = millis();
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t)) cachedTemp = t;
    if (!isnan(h)) cachedHum  = h;
  }

  bool warm = millis() > WARMUP_MS;
  Serial.printf("[%s] MQ135=%.0f mV  T=%.1f  RH=%.0f  %s\n", NODE_ID, mv, cachedTemp, cachedHum, warm ? "" : "(warming up)");
  if (!warm || WiFi.status() != WL_CONNECTED) return;

  char body[200];
  snprintf(body, sizeof(body),
           "{\"node\":\"%s\",\"mv\":%.1f,\"temp\":%.1f,\"hum\":%.0f,\"rssi\":%d}",
           NODE_ID, mv, cachedTemp, cachedHum, WiFi.RSSI());

  HTTPClient http;
  http.setReuse(true); // Re-use HTTP connection for fast throughput
  http.setTimeout(800);
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  if (code != 200 && code != 204) Serial.printf("POST failed: %d\n", code);
  http.end();
}
