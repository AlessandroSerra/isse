# `convert_file.py` — Documentazione

Script Python per la conversione di file di struttura atomica tra i formati usati dai principali codici DFT e MD (VASP, Quantum ESPRESSO, LAMMPS, ALAMode, CIF, extXYZ, …), con suite di test di equivalenza strutturale eseguita automaticamente dopo ogni conversione.

---

## Dipendenze

| Libreria | Ruolo | Obbligatoria |
|---|---|---|
| `ase` | I/O di tutti i formati, manipolazione della struttura | Sì |
| `numpy` | Algebra lineare, confronti numerici | Sì |
| `scipy` | KDTree per il test sulle coordinate frazionarie | Sì |
| `spglib` | Determinazione del gruppo spaziale | No (test 5 saltato) |

---

## Utilizzo

```
convert_file.py <input> [<input> ...] [-o OUTPUT] [-it FORMAT] [-ot FORMAT]
                [-r nx ny nz] [--skip-tests] [--list-formats]
```

### Argomenti posizionali

| Argomento | Descrizione |
|---|---|
| `input` | Uno o più file di input, oppure pattern glob (`"*.xyz"`) |

### Opzioni

| Flag | Descrizione |
|---|---|
| `-o`, `--output` | File di output (solo con singolo file in ingresso) |
| `-it`, `--input-type` | Forza il formato di input (stringa ASE, es. `vasp`, `extxyz`) |
| `-ot`, `--output-type` | Forza il formato di output; obbligatorio in modalità batch |
| `-r nx ny nz`, `--replicate` | Costruisce un supercell replicando la struttura lungo x, y, z |
| `--skip-tests` | Disabilita i test di equivalenza dopo la conversione |
| `--list-formats` | Stampa tutti i formati ASE disponibili ed esce |

### Esempi

```bash
# Conversione singola: extXYZ → VASP
convert_file.py structure.xyz -o POSCAR -it extxyz -ot vasp

# Supercell 2×2×2: extXYZ → VASP
convert_file.py model.relaxed -o super.vasp -it extxyz -ot vasp -r 2 2 2

# Batch: tutti i POSCAR in una cartella → extXYZ
convert_file.py "POSCAR_*" -ot extxyz

# Senza test (utile in pipeline veloci)
convert_file.py structure.xyz -o out.lmp --skip-tests

# Lista formati supportati
convert_file.py --list-formats
```

### Comportamento del formato automatico

Se `-it` o `-ot` vengono omessi, il formato viene dedotto dal nome file:

| Pattern nel nome | Formato dedotto |
|---|---|
| `POSCAR`, `CONTCAR` | `vasp` |
| `*.xyz` | `extxyz` |
| `*.lmp` | `lammps-data` |
| `*.cif` | `cif` |
| `*.pwi` | `espresso-in` |
| `*.pwo` | `espresso-out` |

---

## Architettura dello script

Lo script è organizzato in tre strati indipendenti.

```
┌─────────────────────────────────────────────┐
│                    CLI (main)               │
│  parsing argomenti, loop su coppie file     │
└────────────────────┬────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
┌─────────▼──────────┐  ┌──────▼────────────────┐
│    convert()       │  │  run_equivalence_tests │
│  lettura, replica, │  │  7 check strutturali   │
│  sort, scrittura   │  │  su input vs output    │
└────────────────────┘  └───────────────────────┘
```

Le funzioni di test sono pure (nessun side-effect, nessuna dipendenza dallo stato della conversione) e possono essere importate separatamente:

```python
from convert_file import convert, run_equivalence_tests
```

---

## Logica di conversione

### Lettura

Per i formati standard (`extxyz`, `vasp`, `cif`, `espresso-in/out`, `lammps-data`, …) la lettura è delegata ad `ase.io.read`. Per il formato ALAMode/LAMMPS dump (`alm.lmp`, `lammpstrj`) viene usato un parser custom (`_read_lammps_alamode`) perché il file non rispetta il formato LAMMPS dump standard che ASE si aspetta: la sezione `BOX BOUNDS` contiene i vettori di cella interi (3×3) invece dei valori `xlo xhi`, `ylo yhi`, `zlo zhi` convenzionali.

### Supercell (`-r nx ny nz`)

La replica viene eseguita con `ase.Atoms.repeat([nx, ny, nz])` prima della scrittura. Dopo la replica si applica `wrap(eps=1e-12)` per ricondurre eventuali atomi fuori dai bordi PBC all'interno della cella, usando una soglia molto piccola per non spostare atomi che si trovano legittimamente sul bordo.

### Ordinamento per VASP

Il formato POSCAR richiede che tutti gli atomi della stessa specie siano raggruppati consecutivamente. Prima di scrivere file VASP viene quindi applicato `ase.build.sort`, che riordina gli atomi per numero atomico. Questo riordinamento è la ragione principale per cui il test di allineamento rigido (Kabsch) viene sistematicamente saltato nelle conversioni verso VASP.

### Scrittura

| Formato | Metodo |
|---|---|
| `lammps-data` | `ase.io.lammpsdata.write_lammps_data(..., masses=True)` |
| `alm.lmp`, `lammpstrj` | `_write_lammps_alamode` (custom) |
| `extxyz` | `ase.io.write` con colonne `symbols`, `positions`, `masses` |
| Tutti gli altri | `ase.io.write(..., direct=True)` (coordinate frazionarie) |

Per `extxyz`, la colonna `masses` viene popolata esplicitamente con `set_masses(get_masses())`: un'operazione apparentemente ridondante ma necessaria affinché ASE includa la colonna nel file scritto.

---

## Suite di test di equivalenza

I test vengono eseguiti **dopo** che il file è stato scritto su disco, rileggendolo con `ase.io.read`. Questo garantisce che si stia verificando la struttura effettivamente salvata, non quella in memoria. La struttura di riferimento è `orig.repeat([nx,ny,nz])` quando si usa `--replicate`, oppure `orig` direttamente.

### Test 1 — Conteggio atomi

**Cosa controlla:** `len(ref) == len(conv)`

**Perché:** È il controllo più elementare e il più rapido da fallire. Con `--replicate`, il valore atteso viene calcolato come `N_input × nx × ny × nz` invece di confrontare direttamente con l'input non replicato.

**Quando fallisce (casi reali):** Formato di output che non supporta strutture periodiche e tronca gli atomi fuori dalla cella; bug nel parser custom ALAMode se la riga `NUMBER OF ATOMS` viene scritta male.

---

### Test 2 — Composizione chimica

**Cosa controlla:** `Counter(ref.symbols) == Counter(conv.symbols)`

**Perché:** Un errore di conversione comune è la perdita o la sostituzione di specie chimiche, specialmente quando il formato di output non codifica le specie esplicitamente (es. LAMMPS data file con tipi numerici `1`, `2`, … invece di simboli). Il confronto usa `Counter` invece di liste ordinate per essere insensibile all'ordinamento degli atomi. Il messaggio di errore mostra solo le differenze (`{'B': 1600}` invece di una lista di 3200 elementi).

**Quando fallisce (casi reali):** Conversione verso formati che usano tipi numerici (`lammps-data`) se la mappatura tipo→specie non viene preservata; rilettura di file VASP senza la riga degli elementi nel POSCAR.

---

### Test 3 — Parametri di cella (a, b, c, α, β, γ)

**Cosa controlla:** `allclose(ref.cell.cellpar(), conv.cell.cellpar(), atol=1e-4)`

I parametri confrontati sono le lunghezze dei tre vettori di reticolo in Å e i tre angoli in gradi.

**Perché:** La cella è la quantità più critica in un calcolo DFT/MD: determina la pressione, il volume, la struttura elettronica. Anche una piccola variazione in un parametro può cambiare il comportamento fisico del sistema. La tolleranza `1e-4 Å` è un ordine di grandezza più stretta della precisione tipica dei calcoli DFT (convergenza a ~`1e-3 Å`).

**Quando fallisce (casi reali):** Formato che arrotonda la cella a poche cifre decimali; conversione tra rappresentazioni ortogonali e tricliniche che introduce errori di troncamento.

---

### Test 4 — Forma della cella (riduzione di Minkowski)

**Cosa controlla:** Equivalenza tra le celle ridotte di Minkowski delle due strutture.

La riduzione di Minkowski trova la rappresentazione "più corta" possibile di una cella, invariante per trasformazioni unitmodulari (operazioni $GL(3,\mathbb{Z})$). Due celle che descrivono lo stesso reticolo ma con vettori scelti diversamente (es. a causa di una diversa convenzione del formato) hanno la stessa riduzione di Minkowski.

**Perché:** Alcuni formati scelgono una base diversa per gli stessi vettori di reticolo. Per esempio, CIF e VASP possono usare convenzioni diverse per celle tricliniche. Il test sui parametri di cella (test 3) fallirebbe in questo caso anche se le strutture sono fisicamente identiche; il test di Minkowski è invariante rispetto a questa scelta.

**Nota implementativa:** `ase.geometry.minkowski_reduce` restituisce un `np.ndarray` grezzo (matrice 3×3), non un oggetto `Cell`. È necessario wrapparlo con `ase.cell.Cell(...)` prima di chiamare `.cellpar()`.

**Quando fallisce (casi reali):** Conversione che altera effettivamente i vettori di reticolo, non solo la loro rappresentazione.

---

### Test 5 — Gruppo spaziale e volume (spglib)

**Cosa controlla:**
- `dataset1.number == dataset2.number` (numero di gruppo spaziale internazionale)
- `isclose(vol1, vol2, rtol=1e-5)` (volume della cella)

**Perché:** Il gruppo spaziale è la firma della simmetria cristallografica. Se una conversione altera le posizioni atomiche in modo tale da rompere o introdurre simmetrie, spglib lo rileva confrontando i dataset di simmetria. È il test più potente per strutture cristalline ordinate. Il volume è incluso perché due strutture possono avere lo stesso gruppo ma celle di dimensioni diverse.

**Dipendenza:** `spglib`. Se non installato il test viene saltato con un avviso, senza interrompere lo script.

**Quando fallisce (casi reali):** Errata replica del supercell che introduce discontinuità; errori di PBC wrap che spostano atomi sul bordo in posizioni simmetricamente inequivalenti.

---

### Test 6 — Coordinate frazionarie (KDTree, PBC-aware)

**Cosa controlla:** Per ogni atomo di specie Z in `ref`, verifica che esista un atomo della stessa specie in `conv` a distanza frazionaria < `1e-3`.

**Perché è il test più critico:** È l'unico che confronta le posizioni atomiche individuali. Tutti gli altri test verificano proprietà globali (cella, simmetria, composizione) che possono rimanere corrette anche se qualche atomo si è spostato. Questo test cattura errori locali.

**Perché KDTree e non `lexsort`:**

Un approccio naive ordina entrambe le strutture per posizione e confronta elemento per elemento. Questo fallisce in due scenari frequenti:

1. **Posizioni quasi-degeneri:** in strutture rilassate, due atomi della stessa specie possono avere coordinate frazionarie quasi identiche (es. `x=0.2500` e `x=0.2501`). Una minuscola differenza numerica introdotta dalla scrittura/rilettura può invertirne l'ordine nel `lexsort`, causando un'associazione sbagliata con `max Δ ≈ 0.5` (distanza al vicino sbagliato) invece di `max Δ ≈ 1e-4` (differenza reale).

2. **Bordi PBC:** un atomo a coordinata `0.9999` e uno a `0.0001` rappresentano la stessa posizione a meno di un vettore di reticolo. Dopo `% 1.0` la differenza appare `≈ 1` invece di `≈ 0`.

La soluzione adottata usa `scipy.spatial.KDTree` con `boxsize=np.ones(3)`, che:
- Effettua il nearest-neighbor search direttamente in coordinate frazionarie
- Gestisce la periodicità in modo nativo (`boxsize` attiva il fold-over PBC)
- È separato per specie, quindi B è confrontato solo con B e N solo con N
- Ha complessità O(N log N) invece di O(N²) bruto

**Output diagnostico:** il test riporta sempre `max Δ = X.XXe-YY` per distinguere un vero errore di conversione (Δ ~ unità) da un falso positivo numerico (Δ ~ 1e-15).

---

### Test 7 — Allineamento rigido (Kabsch)

**Cosa controlla:** Dopo aver minimizzato rotazione e traslazione tra le due strutture (algoritmo di Kabsch via `ase.build.minimize_rotation_and_translation`), verifica che le posizioni cartesiane e la cella differiscano di meno di `1e-4 Å`.

**Perché:** Cattura il caso in cui la struttura è fisicamente corretta ma orientata diversamente nello spazio. Questo può accadere con formati che non fissano la convenienza dell'asse (es. CIF con setting non standard).

**Quando viene saltato (SKIP):** Quando `sort()` ha riordinato gli atomi o `wrap()` ha spostato atomi attraverso il bordo PBC, l'algoritmo di Kabsch non può funzionare perché richiede una corrispondenza posizione-per-posizione nell'ordine originale. In questo caso il test viene saltato automaticamente con un rimando al test 6, che è robusto a questi riordinamenti.

---

## Tolleranze di confronto

| Test | Tolleranza | Grandezza fisica |
|---|---|---|
| Parametri di cella | `1e-4 Å` / `1e-4°` | ~100× la convergenza DFT tipica |
| Riduzione Minkowski | `1e-5` | Confronto adimensionale sui parametri ridotti |
| Coordinate frazionarie | `1e-3` (frazionale) | ~0.05 Å su celle da 50 Å |
| Allineamento Kabsch | `1e-4 Å` | Coerente con i parametri di cella |

La tolleranza del test 6 è più larga delle altre perché il formato POSCAR scrive posizioni con 6 cifre decimali, che su celle grandi (~50 Å) corrispondono a errori di arrotondamento fino a ~5×10⁻⁴ in coordinate frazionarie.

---

## Exit code

Lo script termina con exit code `1` se almeno un test fallisce su almeno una coppia di file, con `0` altrimenti. Questo permette di usarlo in pipeline CI/CD:

```bash
convert_file.py POSCAR -o out.xyz && echo "OK" || echo "CONVERSIONE FALLITA"
```
