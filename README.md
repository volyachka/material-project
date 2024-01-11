# Exploring materialscloud:2019.0077/v1

This repo contains code for exploring the data shared along with the paper [doi:10.1039/C9EE02457C](https://dx.doi.org/10.1039/C9EE02457C). The actual data is located [here](https://archive.materialscloud.org/record/2019.0077/v1).

## Preparation steps

**Step 1**. Download the AiiDA archive from [materialscloud](https://archive.materialscloud.org/record/2019.0077/v1).

**Step 2**. Install AiiDA:

```bash
pip install aiida
```

**Step 3**. Update the archive:

```bash
verdi archive migrate screening.aiida migrated.aiida
```
Note: this doubles the amount of disk space used. There's some configuration of the above command that does this inplace to save space (check out the docs).
# material-project
