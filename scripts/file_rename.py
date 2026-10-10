import argparse
import sys
from pathlib import Path

def rinomina_file(cartella: str, prefisso: str, start_index: int, cifre: int = 3):
    path = Path(cartella)

    if not path.is_dir():
        print(f"Errore: Il percorso '{cartella}' non esiste o non è una cartella.", file=sys.stderr)
        sys.exit(1)

    # Prendi solo i file, escludendo eventuali sotto-cartelle
    file_list = [f for f in sorted(path.iterdir()) if f.is_file()]

    if not file_list:
        print(f"Nessun file trovato in: {cartella}")
        return

    conteggio = start_index
    for file in file_list:
        # file.suffix preserva l'estensione originale (es. .jpg, .csv, .png)
        nuovo_nome = f"{prefisso}{conteggio:0{cifre}d}{file.suffix}"
        destinazione = path / nuovo_nome

        # Evita di sovrascrivere se il file ha già il nome corretto
        if file != destinazione:
            file.rename(destinazione)
            print(f"Rinominato: {file.name} -> {nuovo_nome}")
        else:
            print(f"Saltato (già nominato): {file.name}")

        conteggio += 1

    print(f"\nOperazione completata! Rinominati file fino al conteggio: {conteggio - 1}")

def main():
    parser = argparse.ArgumentParser(
        description="Rinomina tutti i file in una cartella con un prefisso e un indice progressivo, preservando l'estensione originale."
    )
    
    # Argomenti posizionali o opzionali
    parser.add_argument("cartella", type=str, help="Percorso della cartella contenente i file")
    parser.add_argument("-p", "--prefisso", type=str, default="", help="Prefisso per i file (es. 'img_', 'dati_')")
    parser.add_argument("-s", "--start", type=int, default=1, help="Numero di partenza del conteggio (default: 1)")
    parser.add_argument("-c", "--cifre", type=int, default=3, help="Cifre minime per il contatore con padding di zeri (default: 3, es. 001)")

    args = parser.parse_args()

    rinomina_file(
        cartella=args.cartella,
        prefisso=args.prefisso,
        start_index=args.start,
        cifre=args.cifre
    )

if __name__ == "__main__":
    main()