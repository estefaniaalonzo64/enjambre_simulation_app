# CDMX — Rain fill & Desazolve simulator (Streamlit)

Interactive simulator with a **bundled default dataset**:
`data/atlas-de-riesgo-inundaciones.csv`.

- Upload your own CSV or run with the bundled atlas.
- Tune parameters and visualize **baseline**, **desazolve**, and **impact** maps.
- Export static PNGs and an **animation** (GIF and MP4 fallback).

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

## Deployment

### Streamlit Community Cloud (free)
1. Push this repo to GitHub.
2. Create an app from the repo, set the main file to `app.py`.
3. Done. The bundled dataset is used by default.

### Docker
```bash
docker build -t cdmx-flood .
docker run --rm -p 8501:8501 cdmx-flood
```

## Data format (expected)
CSV columns:
- `alcaldia` (str)
- `int2` (int, 1..5)
- `area_m2` (float)
- `geo_shape` (GeoJSON string of a `Polygon` with outer ring coordinates)

> The app will skip malformed rows and will raise if no Polygon is found.

## Why both GIF and MP4?
Some environments (e.g., Windows Explorer preview / Photos) don’t animate GIFs
reliably. The app writes both formats; **MP4 (H.264)** plays almost everywhere.

## File tree
```
.
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ run.sh
├─ run.bat
├─ .streamlit/
│  └─ config.toml
├─ data/
│  └─ atlas-de-riesgo-inundaciones.csv
└─ README.md
```

## License
MIT or your preferred license.
