from pathlib import Path
import argparse


TURKISH_LOWER_MAP = str.maketrans({
    "I": "ı",
    "İ": "i",
})


def turkish_lower(text: str) -> str:
    return text.translate(TURKISH_LOWER_MAP).lower()


def temizle_sokak_isimleri(lines: list[str]) -> list[str]:
    seen = set()
    sonuc = []

    for line in lines:
        isim = " ".join(turkish_lower(line).split())
        if isim and isim not in seen:
            seen.add(isim)
            sonuc.append(isim)

    return sonuc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sokak isimlerini küçültür, boşlukları düzenler ve tekrarları siler."
    )
    parser.add_argument(
        "input", nargs="?", help="Girdi dosyası"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Çıktı dosyası",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent
    input_path = Path(args.input) if args.input else data_dir / "sokak_isimleri.txt"
    output_path = (
        Path(args.output)
        if args.output
        else data_dir / "temiz_sokak_isimleri.txt"
    )

    lines = input_path.read_text(encoding="utf-8").splitlines()
    sonuc = temizle_sokak_isimleri(lines)
    output_path.write_text("\n".join(sonuc) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
