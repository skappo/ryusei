import os
import urllib.request
import tetra3

# ---------------------------------------------------------
# 1. Locate Tetra3
# ---------------------------------------------------------
t3_dir = os.path.dirname(tetra3.__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(script_dir, "assets", "db")

os.makedirs(db_dir, exist_ok=True)

print(f"Tetra3 directory: {t3_dir}")
print(f"Database directory: {db_dir}")

# ---------------------------------------------------------
# 2. Download Hipparcos catalog if necessary
# ---------------------------------------------------------
hip_path = os.path.join(db_dir, "hip_main.dat")

if not os.path.exists(hip_path):
    print(f"\nHipparcos catalog missing.")
    print(f"Downloading to: {hip_path}")
    print("This may take a while...")

    url = "https://cdsarc.cds.unistra.fr/ftp/I/239/hip_main.dat"
    urllib.request.urlretrieve(url, hip_path)

    print("Download complete!")

else:
    print(f"\nHipparcos catalog found:")
    print(hip_path)

# ---------------------------------------------------------
# 3. Create Tetra3 object
# ---------------------------------------------------------
print("\nInitializing Tetra3...")

t3 = tetra3.Tetra3(load_database=None)

# ---------------------------------------------------------
# 4. Generate database
# ---------------------------------------------------------
db_name = os.path.join(db_dir, "default_database")

print("\nGenerating Tetra3 database...")
print("Maximum FOV: 65 degrees")
print("This can take several minutes on a Raspberry Pi.")

t3.generate_database(
    max_fov=65,
    save_as=db_name,
    star_catalog="hip_main"
)

print("\nDatabase generation complete!")
print(f"Database: {db_name}.npz")
