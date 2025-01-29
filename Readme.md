# Relax-Bench 🪑
---
##### Jan 29, 2025 | S.Alireza Hashemi

Github:
[Link](https://github.com/salireza111/AR_PDB_Visualizer)



<div style="text-align: justify"> 
Ever dreamed of touching your protein? Now you can—virtually—with AR_PDB Viewer!
</div>


![scene](./Demo.webp)


##### Change the path for your designated PDB

```
if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdb_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdb_path = script_dir + "/RNR.pdb" #Place your path here

```

 
### Installation guide:

```bash
# Git clone
git clone https://github.com/salireza111/AR_PDB_Visualizer.git
cd AR_PDB_Visualizer

# Create the virtual env 
python3 -m venv venv

# Activate the environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the code
python main.py
```
