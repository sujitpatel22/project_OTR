import os
from pathlib import Path
import shutil

for file in os.listdir("labels-xml"):
    label_class = file.split(".")[0].split("-")[0]
    folder_path = Path(f"labels-xml/{label_class}")
    if not folder_path.exists():
        folder_path.mkdir()
    source_path = "labels-xml/"+file
    destination_path = "labels-xml/"+label_class+"/"+file
    shutil.move(source_path, destination_path)