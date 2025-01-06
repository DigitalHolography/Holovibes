import os
import time
import sys

file_to_check = sys.argv[1]  # Nom du fichier en argument
max_wait = 60  # secondes
wait_interval = 0.5  # secondes


def is_file_locked(filepath):
    """Retourne True si le fichier est verrouillé."""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "a"):
            return False
    except IOError:
        return True


start_time = time.time()
while is_file_locked(file_to_check):
    if time.time() - start_time > max_wait:
        print(f"Error: File {file_to_check} still locked after {max_wait} seconds!")
        sys.exit(1)
    print(f"Waiting for {file_to_check} to be unlocked...")
    time.sleep(wait_interval)

print(f"File {file_to_check} is accessible. Proceeding...")
sys.exit(0)
