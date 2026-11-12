"""Download the public USDA sweetpotato chip composition/quality CSV.
Run from the project root: python scripts/download_usda_sweetpotato_dataset.py
"""
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://ndownloader.figshare.com/files/64495524"
OUT = Path(__file__).resolve().parents[1] / "data" / "real_data" / "raw" / "usda_sweetpotato_chip_texture_fat.csv"

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "VirtualChipFactoryResearchPrototype/1.0"})
    try:
        with urlopen(request, timeout=45) as response:
            payload = response.read()
        if not payload or b"<html" in payload[:500].lower():
            raise RuntimeError("The server did not return a CSV payload. Download it manually from the source page.")
        OUT.write_bytes(payload)
        print(f"Downloaded {len(payload):,} bytes to {OUT}")
        print("Source: https://catalog.data.gov/dataset/data-from-sweetpotato-chip-texture-and-fat-content-and-the-impact-of-composition-and-glass")
        print("Inspect the header and metadata before modeling; this is sweetpotato data, not ordinary potato-chip data.")
    except Exception as exc:
        print(f"Automatic download failed: {exc}")
        print("Manual fallback: open the Data.gov source page, click Download, and save the CSV as:")
        print(OUT)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
