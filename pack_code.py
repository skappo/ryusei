import os

with open("meteora_codebase.txt", "w", encoding="utf-8") as outfile:
    for filename in sorted(os.listdir(".")):
        if filename.endswith(".py") and filename != "pack_code.py":
            outfile.write(f"\n\n{'='*60}\n")
            outfile.write(f"--- FILE: {filename} ---\n")
            outfile.write(f"{'='*60}\n\n")
            with open(filename, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())

print("Created meteora_codebase.txt!")
