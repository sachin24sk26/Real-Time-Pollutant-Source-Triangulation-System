# Real-Time Pollutant Source Triangulation System - Build Guide

3 nodes (ESP32 + MQ-135 + DHT11) --Wi-Fi--> Flask server on your laptop (localhost)
--> centroid + trilateration + AI --> live dashboard at http://localhost:5000

```
triangulation_project/
  node_firmware/node_firmware.ino   <- flash on all 3 ESP32 (change NODE_ID only)
  server/
    app.py              Flask server + API (the "localhost")
    localize.py         all formulas: centroid, trilateration, AI features
    train.py            trains Random Forest (+ fits path-loss exponent from real data)
    simulate_nodes.py   fake 3 nodes, test everything with NO hardware
    test_accuracy.py    Monte-Carlo accuracy test for your report
    config.json         node positions (metres) + parameters  <-- EDIT THIS
    static/index.html   live dashboard
```

---------------------------------------------------------------------------
## STEP 1 - Wire each node (x3)

| Part | Pin | Goes to |
|---|---|---|
| MQ-135 | VCC | ESP32 **VIN (5 V)** |
| MQ-135 | GND | GND |
| MQ-135 | AO | **10 kOhm -> GPIO34 -> 15 kOhm -> GND** (voltage divider) |
| DHT11 | VCC / GND | 3V3 / GND |
| DHT11 | DATA | GPIO4 (10 kOhm pull-up to 3V3 if your module has no resistor) |

**Do not skip the divider.** MQ-135 runs on 5 V so AO can reach ~5 V, but an ESP32 pin
must never see more than 3.3 V. 10k/15k scales by 0.6 (5 V -> 3.0 V) and the firmware
undoes the 0.6 factor. GPIO34 is used because it is on ADC1 (ADC2 stops working with Wi-Fi).
Power each node from a USB power bank (MQ heater needs 5 V, ~150 mA).

## STEP 2 - Flash the ESP32s
1. Arduino IDE -> Boards Manager -> install **esp32 by Espressif**. Board: *ESP32 Dev Module*.
2. Library Manager -> install **DHT sensor library (Adafruit)** (+ Adafruit Unified Sensor).
3. Open `node_firmware.ino`, edit: `NODE_ID` ("A"/"B"/"C"), `WIFI_SSID`, `WIFI_PASS`, `SERVER_URL`.
4. Upload to each ESP32 with its own NODE_ID. Open Serial Monitor (115200): you should see
   `[A] MQ135=... mV T=.. RH=..`. Nodes stay quiet for 60 s (MQ warm-up) before sending.

## STEP 3 - Laptop server
```bash
cd server
pip install -r requirements.txt
python train.py          # trains the AI model (takes ~20 s)
python app.py            # starts server on port 5000
```
* Find your laptop IP: Windows `ipconfig` (IPv4), Mac/Linux `ifconfig` / `ip a`. Put it in `SERVER_URL`
  (e.g. `http://192.168.1.50:5000/api/reading`). Laptop and ESP32s must be on the **same 2.4 GHz network**.
* Windows: allow Python through the firewall (Private networks) or the ESP32s can't reach you.
* College Wi-Fi often blocks device-to-device traffic ("client isolation"). **Use your phone's hotspot (2.4 GHz)**
  - connect laptop + all 3 ESP32s to it.
* Open http://localhost:5000 - nodes turn green when they report.

**No hardware yet?** Test everything first: `python app.py` in one terminal, then
`python simulate_nodes.py --move` in another. A moving source appears on the map.

## STEP 4 - Place the nodes and edit `config.json`
* Put the 3 nodes ~3-4 m apart in a triangle (smaller than a big hall - gas fades fast), sensors at the same height.
* Pick node A as origin (0, 0), x = east, y = north. **Measure with a tape** and write the real positions:
  ```json
  "A": {"x": 0.0, "y": 0.0}, "B": {"x": 4.0, "y": 0.0}, "C": {"x": 2.0, "y": 3.46}
  ```
  (An equilateral triangle of side 4 m has C = (2, 3.46).) Restart `app.py` after editing.
* Optional: put node A's real GPS in `origin_lat/origin_lng` to get lat/lng output.

## STEP 5 - Calibrate (clean air)
1. Let all nodes run **10+ minutes** in clean air (MQ-135 needs a warm-up; readings drift while it settles).
2. Dashboard -> **Set baseline**. This stores each node's clean-air voltage as "zero". Every node has a
   different baseline, so this step is required. Signal = reading - baseline.
3. Re-do the baseline if you move the nodes or the room temperature changes a lot.

## STEP 6 - Test and train on REAL data (this is what makes the AI honest)
Safe test sources: an **agarbatti (incense) stick** or a cotton ball with a few drops of hand sanitizer.
Keep anything flammable/liquid away from the electronics.
1. Mark ~15-25 spots inside and around the triangle with tape; measure their (x, y).
2. Put the source on a spot, wait ~20-30 s for the readings to settle, type x,y in the dashboard -> **Log sample**.
3. Repeat for all spots (air the room between spots and re-check the signals fall back to ~0).
4. Stop the server, run `python train.py` again. It now (a) **fits the path-loss exponent n** to your room and
   (b) blends your real samples into the Random Forest. Restart `app.py`.
5. Test on NEW spots you didn't log and note the error (|estimate - true|) - put this table in your report.

## STEP 7 - The maths (for report / viva)

Signal each node sees from a source at distance d:  **s_i = A / d_i^n**   (A = strength, n ~ 2 in open air)

1. **Weighted centroid** (the baseline): x = sum(s_i * x_i) / sum(s_i), y = sum(s_i * y_i) / sum(s_i)
2. **Trilateration** (distance-based): take logs, ln s_i = ln A - n ln d_i with
   d_i = sqrt((x - x_i)^2 + (y - y_i)^2). Solve for (x, y, ln A) by non-linear least squares (SciPy
   `least_squares`), i.e. find the point whose 3 distance circles best intersect. The unknown strength A
   is solved together with the position, so no per-node distance calibration is needed. A 4th node
   gives an over-determined system = more accurate.
3. **AI**: Random Forest regressor, input = [share of total signal per node, log(total)], output = (x, y).
   Pre-trained on 20,000 simulated plumes, then fine-tuned with your real logged samples.
4. **Final** = average of trilateration and AI (falls back to whichever is available).

**Terminology for your viva:** strictly, using *distances* is **trilateration**; *triangulation* uses angles.
Your project estimates distances from concentration, so say *"concentration-based trilateration
(often loosely called triangulation), with weighted centroid and an ML model for comparison"*. Use the same
wording on the PPT (slide 3 says "Trilateration", the title says "Triangulation" - make them consistent).

## Accuracy on SIMULATED data (500 random sources, 4 m triangle, 15% + 4 mV noise)
These numbers come from the simulator, **not real hardware** - real MQ-135 readings are noisier and
slower, so expect worse. Use them only to show the code works; report your own measured error.

| Scenario | Centroid | Trilateration | AI | Final |
|---|---|---|---|---|
| Source inside triangle (mean error, m) | 0.34 | 0.29 | 0.53 | 0.39 |
| Source anywhere in area incl. outside | 1.05 | 0.75 | 0.59 | 0.64 |
| Inside triangle, real n = 1.5 (model mismatch) | 0.29 | 0.33 | 0.37 | 0.32 |

Takeaway: inside the triangle the simple centroid is already close (it can never point outside the
triangle); trilateration/AI matter when the source is outside it or the physics differs - that is
why Step 6 (fit n + real samples) is important.

## Troubleshooting
| Problem | Fix |
|---|---|
| ESP32 prints `POST failed: -1` | Wrong IP, firewall, different network, or Wi-Fi client isolation -> use phone hotspot |
| Node stays red on dashboard | Still in 60 s warm-up, or `SERVER_URL` wrong |
| MQ reading always ~0 or max | Check divider wiring and that AO (not DO) is used |
| Readings keep drifting upward | Heater not warmed up; wait 10+ min and re-set baseline |
| "Waiting for node(s) ... (set baseline?)" | Press Set baseline with all nodes online |
| Estimate jumps around | Let air settle; widen smoothing (`smooth_window_seconds`) in config.json |
| Estimate always near one node | Nodes at different heights / one sensor weaker -> compare baselines, re-calibrate |

## Safety
Use small smoke/alcohol sources only, ventilated room, no open flame near the electronics.
