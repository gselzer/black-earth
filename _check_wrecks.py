import requests, json

query = """
[out:json];
(
  node["seamark:type"="wreck"](45.1,-87.3,45.5,-86.7);
  node["historic"="wreck"](45.1,-87.3,45.5,-86.7);
  node["seamark:type"="obstruction"](45.1,-87.3,45.5,-86.7);
);
out body;
"""

resp = requests.post(
    "https://overpass.openstreetmap.fr/api/interpreter",
    data={"data": query},
    headers={"User-Agent": "black-earth-wallpaper/1.0"},
    timeout=60,
)
print("HTTP", resp.status_code)
data = resp.json()
print(f"Found {len(data['elements'])} features")
for el in data["elements"]:
    lon = el["lon"] if el["type"] == "node" else el["center"]["lon"]
    lat = el["lat"] if el["type"] == "node" else el["center"]["lat"]
    name = el.get("tags", {}).get("name", "(unnamed)")
    print(f"  {el['type']:4s}  {lat:.4f}, {lon:.4f}  {name}")
