import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font


def _font_with(font, **overrides):
    """
    Return a new Font equal to `font` with `overrides` applied.

    Replaces the deprecated Font.copy(**kwargs). The tempting alternative,
    `font + Font(**overrides)`, is actually WRONG for this: openpyxl's
    Font.__add__ resolves each attribute as `self.attr or other.attr`, so
    the *existing* font wins whenever its value is truthy -- and Font.color
    is never falsy (even an "unset" font carries a Color(theme=...) object),
    so a color override would silently be dropped. Confirmed by round-
    tripping a saved/reloaded workbook: the override never took effect.
    Building the kwargs explicitly sidesteps that truthy/falsy ambiguity.
    """
    attrs = {
        "name": font.name,
        "size": font.size,
        "bold": font.bold,
        "italic": font.italic,
        "vertAlign": font.vertAlign,
        "underline": font.underline,
        "strike": font.strike,
        "color": font.color,
    }
    attrs.update(overrides)
    return Font(**attrs)


# ============================================================
# CONFIGURAZIONE
# ============================================================

# File di input
INPUT_FILE = "/media/pi/USB/logs/pipeline.log"

# File Excel di output
OUTPUT_FILE = "/media/pi/USB/logs/meteora_log.xlsx"


# ============================================================
# REGEX
# ============================================================

# Esempio:
# 2026-09-07 19:12:45,499 [INFO] [MainThread] Initialization successful.

LOG_PATTERN = re.compile(
    r"^(?P<data>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<ora>\d{2}:\d{2}:\d{2},\d{3})\s+"
    r"\[(?P<lvl>[A-Z]+)\]\s+"
    r"\[(?P<thread>[^\]]+)\]\s*"
    r"(?P<descrizione>.*)$"
)


# ============================================================
# LETTURA LOG
# ============================================================

def convert_log_to_excel(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"ERRORE: file non trovato: {input_path}")
        return False

    records = []
    invalid_lines = []

    # encoding UTF-8 con gestione caratteri problematici
    with open(input_path, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            match = LOG_PATTERN.match(line)

            if match:
                records.append({
                    "data": match.group("data"),
                    "ora": match.group("ora"),
                    "lvl": match.group("lvl"),
                    "thread": match.group("thread"),
                    "descrizione": match.group("descrizione"),
                })
            else:
                # Conserviamo le righe che non rispettano il formato
                invalid_lines.append({
                    "riga": line_number,
                    "contenuto": line
                })

    if not records:
        print("ERRORE: nessuna riga di log valida trovata.")
        return False

    # ========================================================
    # CREAZIONE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        records,
        columns=[
            "data",
            "ora",
            "lvl",
            "thread",
            "descrizione"
        ]
    )

    # ========================================================
    # ESPORTAZIONE EXCEL
    # ========================================================

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl"
    ) as writer:

        # Foglio principale
        df.to_excel(
            writer,
            sheet_name="Log",
            index=False
        )

        # Foglio con eventuali righe non riconosciute
        if invalid_lines:
            invalid_df = pd.DataFrame(invalid_lines)

            invalid_df.to_excel(
                writer,
                sheet_name="Righe_non_riconosciute",
                index=False
            )

        # ====================================================
        # FORMATTAZIONE EXCEL
        # ====================================================

        workbook = writer.book
        worksheet = writer.sheets["Log"]

        # Blocca la prima riga
        worksheet.freeze_panes = "A2"

        # Filtro automatico
        worksheet.auto_filter.ref = worksheet.dimensions

        # Larghezze colonne
        column_widths = {
            "A": 14,   # data
            "B": 16,   # ora
            "C": 12,   # lvl
            "D": 20,   # thread
            "E": 100,  # descrizione
        }

        for column, width in column_widths.items():
            worksheet.column_dimensions[column].width = width

        # Intestazione in grassetto
        # cell.font.copy(bold=True)  # OLD: deprecated in openpyxl
        for cell in worksheet[1]:
            cell.font = _font_with(cell.font, bold=True)

        # Colora il livello del log nella colonna C
        # INFO = verde, WARNING = giallo, ERROR = rosso
        level_colors = {
            "INFO": "008000",     # verde
            "WARNING": "FFD700",  # giallo/oro
            "ERROR": "FF0000",    # rosso
        }

        for row in range(2, worksheet.max_row + 1):
            cell = worksheet[f"C{row}"]
            level = str(cell.value).upper() if cell.value is not None else ""

            if level in level_colors:
                # cell.font.copy(color=level_colors[level], bold=True)  # OLD: deprecated
                cell.font = _font_with(cell.font, color=level_colors[level], bold=True)

    print()
    print("==========================================")
    print(" conversione completata")
    print("==========================================")
    print(f"File input : {input_path}")
    print(f"File output: {output_path}")
    print(f"Righe valide: {len(records)}")
    print(f"Righe non riconosciute: {len(invalid_lines)}")
    print("==========================================")

    if invalid_lines:
        print()
        print(
            "Le righe non riconosciute sono state "
            "salvate nel foglio 'Righe_non_riconosciute'."
        )

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Permette di passare i file dalla riga di comando:
    #
    # python log_to_excel.py mio_log.log risultato.xlsx
    #
    # Se non vengono specificati, usa i nomi configurati sopra.

    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    else:
        input_file = INPUT_FILE

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = OUTPUT_FILE

    convert_log_to_excel(
        input_file,
        output_file
    )
