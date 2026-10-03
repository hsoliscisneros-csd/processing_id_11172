# Data dictionary: anonymised crowdsensed WiFi corpus

Each row is one passive WiFi scan observation of an access point (AP).

| Field | Type | Description |
|:---|:---|:---|
| `bssid` | string | Salted SHA-256 hash of the AP hardware address (irreversible; column name kept for code compatibility). |
| `device_id` | string | Salted SHA-256 hash of the auditor device identifier (irreversible). |
| `security_type` | string | Security class (OPEN / WEP / WPA / WPA2 / WPA3). |
| `security_type_parsed` | string | Normalised security class used in analysis. |
| `level` | integer | RSSI (dBm). |
| `frequency` | integer | Channel centre frequency (MHz). |
| `channel` | integer | IEEE channel. |
| `estimated_distance` | float | Log-distance path-loss range estimate (m). |
| `distance_residual` | float | Observed minus expected distance (m). |
| `vendor` | string | AP hardware vendor from the OUI. |
| `is_randomized_mac` | boolean | Whether the address appears MAC-randomised. |
| `latitude` / `longitude` | float | Auditor coordinates, 3 decimals (~110 m). |
| `municipality` / `state` / `country` / `city` | string | Administrative labels. |
| `timestamp` | integer | Scan time (Unix epoch, ms). |
| `threat_type_heuristic` | string | Heuristic threat-risk label. |
| `is_anomaly_heuristic` | integer | Heuristic binary exposure flag (1 = flagged). |

**Privacy:** raw SSID, BSSID, and device IDs are not included. Values under `bssid` / `device_id` are salted hashes only. Heuristic labels are threat-risk proxies, not adjudicated attacks.

**Citation:** IEEE Latin America Transactions manuscript ID 11172.
